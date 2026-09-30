"""ACLNN binding; public inputs are Q/K/V, public output is normalized O."""
import ctypes as C
from dataclasses import replace


class Operator:
    def __init__(self, library, config):
        self.config = config
        self.scratch = None
        self.buffers = {}
        self.last = None
        self.qualified = set()
        self.library = C.CDLL(str(library), mode=C.RTLD_LOCAL)
        self.acl = C.CDLL('libopapi.so', mode=C.RTLD_LOCAL)
        p, i64p = C.c_void_p, C.POINTER(C.c_int64)
        self.acl.aclCreateTensor.argtypes = [i64p,C.c_uint64,C.c_int,i64p,C.c_int64,C.c_int,i64p,C.c_uint64,p]
        self.acl.aclCreateTensor.restype = p
        self.acl.aclDestroyTensor.argtypes = [p]
        self.acl.aclDestroyTensor.restype = C.c_int
        self.prepare = self.library.aclnnFaFaNkNzVectorFp32V3GetWorkspaceSize
        self.prepare.argtypes = [p,p,p,C.c_int64,C.c_int64,C.c_int64,C.c_int64,C.c_bool,C.c_int64,
                                 p,p,p,p,p,p,C.POINTER(C.c_uint64),C.POINTER(p)]
        self.prepare.restype = C.c_int
        self.launch = self.library.aclnnFaFaNkNzVectorFp32V3
        self.launch.argtypes = [p,C.c_uint64,p,p]
        self.launch.restype = C.c_int

    @staticmethod
    def check(status, where):
        if status:
            raise RuntimeError(f'FaFaNkNzVectorFp32V3 {where}: {status}')

    def allocate(self, q, k, c):
        import torch
        b,h,nq,_ = q.shape
        nk = k.shape[2]//512
        key = tuple(q.shape),tuple(k.shape),q.device,c
        if key not in self.buffers:
            count = b*h*nq*nk*260 if c.capture else 1
            if c.capture and count > 64*1024*1024:
                raise ValueError('debug capture is restricted to small inputs (at most 256 MiB)')
            shapes = dict(sp=(25,3,c.k_block,c.q_block) if c.transpose else (25,3,c.q_block,c.k_block),
                          partial=(25,3,c.q_block,128),trace=(c.groups+1,3,128),stats=(count,),
                          running=(25,c.q_l1,128),output=tuple(q.shape))
            data = {}
            for name,shape in shapes.items():
                dtype = torch.int64 if name=='trace' else torch.float32 if name=='stats' else torch.float16
                data[name] = torch.empty(shape,dtype=dtype,device=q.device)
            data['trace'].fill_(-1)
            if c.groups>25:
                table = torch.tensor([[i,i,0,0] for i in range(25)]+[[25,-1,0,0]],dtype=torch.int64)
                data['trace'][c.groups:].flatten()[:104].copy_(table.flatten().to(q.device))
            self.buffers[key] = data
        return self.buffers[key]

    def _execute(self, q, k, v, c):
        import torch
        import torch_npu
        data = self.allocate(q,k,c)
        ts = (q,k,v,*(data[x] for x in ('sp','partial','trace','stats','running','output')))
        types = [1,1,1,1,1,9,0,1,1]
        handles = []
        try:
            for t, dtype in zip(ts,types):
                dims = (C.c_int64*t.ndim)(*t.shape)
                strides = (C.c_int64*t.ndim)(*t.stride())
                handle = self.acl.aclCreateTensor(dims,t.ndim,dtype,strides,0,2,dims,t.ndim,t.data_ptr())
                if not handle:
                    raise RuntimeError('aclCreateTensor failed')
                handles.append(handle)
            with torch.npu.device(q.device):
                size,executor = C.c_uint64(),C.c_void_p()
                self.check(self.prepare(*handles[:3],c.q_block,512,c.q_l1,c.groups,c.capture,0,
                                       *handles[3:],C.byref(size),C.byref(executor)),'prepare')
                if self.scratch is None or self.scratch.device!=q.device or self.scratch.numel()!=size.value:
                    self.scratch = torch.empty(size.value,dtype=torch.uint8,device=q.device)
                stream = torch.npu.current_stream(q.device)
                raw = torch_npu._C._npu_getCurrentRawStream(q.device.index)
                self.check(self.launch(self.scratch.data_ptr(),size.value,executor,raw),'launch')
                for t in (*ts,self.scratch): t.record_stream(stream)
            self.last = data
            return data['output']
        finally:
            for handle in handles: self.acl.aclDestroyTensor(handle)

    def __call__(self, q, k, v):
        import torch
        if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape:
            raise ValueError('Q/K/V must be [B,H,N,128]')
        b,h,nq,d = q.shape
        if min(q.shape)<=0 or min(k.shape)<=0 or k.shape[0]!=b or d!=128 or k.shape[3]!=128 or h%k.shape[1]:
            raise ValueError('invalid attention geometry')
        c = self.config.resolved(q.shape)
        if nq%c.query_alignment or k.shape[2]%512 or max(nq,k.shape[2])>2**32-1:
            raise ValueError('unsupported query/key tail')
        for t in (q,k,v):
            if t.device.type!='npu' or t.device!=q.device or t.dtype!=torch.float16 or not t.is_contiguous():
                raise ValueError('Q/K/V must be contiguous FP16 on the same NPU')
        if c.groups>25 and q.device not in self.qualified:
            qc = replace(c,groups=25,capture=False)
            probe_q = torch.zeros((1,25,c.q_l1,128),dtype=torch.float16,device=q.device)
            probe_k = torch.zeros((1,1,512,128),dtype=torch.float16,device=q.device)
            probe_v = torch.zeros_like(probe_k)
            result = self._execute(probe_q,probe_k,probe_v,qc)
            ids = self.last['trace'][:25,0,0].cpu().tolist()
            if sorted(ids)!=list(range(25)) or not torch.equal(result,torch.zeros_like(result)):
                raise RuntimeError(f'local 25-Cube MIX qualification failed: {ids}')
            self.qualified.add(q.device)
        return self._execute(q,k,v,c)
