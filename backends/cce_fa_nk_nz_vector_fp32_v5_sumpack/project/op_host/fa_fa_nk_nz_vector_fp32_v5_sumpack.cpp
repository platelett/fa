#include "../op_kernel/variant_config.h"
#include "../op_kernel/fa_fa_nk_nz_vector_fp32_v5_sumpack_tiling.h"
#include "../op_kernel/tiling_key_fa_fa_nk_nz_vector_fp32_v5_sumpack.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
bool Shapes(const gert::Shape &q, const gert::Shape &k, const gert::Shape &v) {
  if (q.GetDimNum() != 4 || k.GetDimNum() != 4 || v.GetDimNum() != 4)
    return false;
  for (int i = 0; i < 4; ++i)
    if (q.GetDim(i) <= 0 || k.GetDim(i) <= 0 || k.GetDim(i) != v.GetDim(i))
      return false;
  return q.GetDim(0) == k.GetDim(0) && q.GetDim(1) % k.GetDim(1) == 0 &&
         q.GetDim(3) == 128 && k.GetDim(3) == 128;
}
bool Geometry(int64_t rows, int64_t cols, int64_t ql, int64_t groups) {
  return rows == (fa_config::WS_Q?fa_config::WS_Q:ql) && cols == fa_config::WS_K && ql == fa_config::Q_L1_LEN && groups >= 1 && groups <= 8192;
}
ge::graphStatus Infer(gert::InferShapeContext *c) {
  auto *a = c->GetAttrs();
  if (!a || !c->GetInputShape(0) || !c->GetInputShape(1) ||
      !c->GetInputShape(2))
    return ge::GRAPH_FAILED;
  const auto &q = *c->GetInputShape(0), &k = *c->GetInputShape(1),
             &v = *c->GetInputShape(2);
  auto *r = a->GetAttrPointer<int64_t>(0), *n = a->GetAttrPointer<int64_t>(1);
  auto *ql = a->GetAttrPointer<int64_t>(2), *g = a->GetAttrPointer<int64_t>(3);
  auto *cap = a->GetAttrPointer<bool>(4);
  auto *phase = a->GetAttrPointer<int64_t>(5);
  if (!phase || *phase != 0 || !r || !n || !ql || !g || !cap || !Shapes(q, k, v) ||
      !Geometry(*r, *n, *ql, *g) || q.GetDim(2) % (fa_config::TRANSPOSE?256:128) || k.GetDim(2) % *n)
    return ge::GRAPH_FAILED;
  *c->GetOutputShape(0) = fa_config::TRANSPOSE ? gert::Shape({25,3,*n,*r}) : gert::Shape({25,3,*r,*n});
  *c->GetOutputShape(1) = gert::Shape({25,3,*r,128});
  *c->GetOutputShape(2) = gert::Shape({*g+1,3,128});
  int64_t debug_size = *cap ? q.GetDim(0)*q.GetDim(1)*q.GetDim(2)*(k.GetDim(2)/512)*260 : 1;
  *c->GetOutputShape(3) = gert::Shape({debug_size});
  *c->GetOutputShape(4) = gert::Shape({25,*ql,128});
  *c->GetOutputShape(5) = q;
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus Types(gert::InferDataTypeContext *c) {
  c->SetOutputDataType(0, ge::DT_FLOAT16);
  c->SetOutputDataType(1, ge::DT_FLOAT16);
  c->SetOutputDataType(2, ge::DT_INT64);
  c->SetOutputDataType(3, ge::DT_FLOAT);
  c->SetOutputDataType(4, ge::DT_FLOAT16);
  c->SetOutputDataType(5, ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus Tile(gert::TilingContext *c) {
  auto *a = c->GetAttrs();
  if (!a || !c->GetPlatformInfo() || !c->GetInputShape(0) ||
      !c->GetInputShape(1) || !c->GetInputShape(2))
    return ge::GRAPH_FAILED;
  const auto &q = c->GetInputShape(0)->GetStorageShape(),
             &k = c->GetInputShape(1)->GetStorageShape();
  const auto &v = c->GetInputShape(2)->GetStorageShape();
  auto *r = a->GetAttrPointer<int64_t>(0), *n = a->GetAttrPointer<int64_t>(1);
  auto *ql = a->GetAttrPointer<int64_t>(2), *g = a->GetAttrPointer<int64_t>(3);
  auto *cap = a->GetAttrPointer<bool>(4);
  auto *phase = a->GetAttrPointer<int64_t>(5);
  if (!phase || *phase != 0 || !r || !n || !ql || !g || !cap || !Shapes(q, k, v) ||
      !Geometry(*r, *n, *ql, *g) || q.GetDim(2) % (fa_config::TRANSPOSE?256:128) || k.GetDim(2) % *n ||
      q.GetDim(2) > UINT32_MAX || k.GetDim(2) > UINT32_MAX ||
      uint64_t(q.GetDim(0)) * q.GetDim(1) * ((q.GetDim(2) + *ql - 1) / *ql) > UINT32_MAX)
    return ge::GRAPH_FAILED;
  auto *d = c->GetTilingData<FaFaNkNzVectorFp32V5SumpackTilingData>();
  if (!d)
    return ge::GRAPH_FAILED;
  *d = {uint32_t(q.GetDim(0)), uint32_t(q.GetDim(1)), uint32_t(k.GetDim(1)),
        uint32_t(q.GetDim(2)), uint32_t(k.GetDim(2)), uint32_t(*ql),
        uint32_t(*g),          uint32_t(*cap), uint32_t(*phase)};
  ASCENDC_TPL_SEL_PARAM(c, uint32_t(*ql / 256 - 1));
  c->SetBlockDim(*g);
  auto *ws = c->GetWorkspaceSizes(1);
  if (!ws)
    return ge::GRAPH_FAILED;
  platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
  uint64_t lib = platform.GetLibApiWorkSpaceSize();
  ws[0] = lib > 64ULL * 1024 * 1024 ? lib : 64ULL * 1024 * 1024;
  // ClearWorkspaceImpl clears 32 KiB per logical AIC plus control records.
  uint64_t logical_scratch=uint64_t(*g)*32768+16ULL*1024*1024;
  if(ws[0]<logical_scratch) ws[0]=logical_scratch;
  return ge::GRAPH_SUCCESS;
}
} // namespace
namespace ops {
class FaFaNkNzVectorFp32V5Sumpack : public OpDef {
public:
  explicit FaFaNkNzVectorFp32V5Sumpack(const char *name) : OpDef(name) {
    this->Input("q")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Input("k")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Input("v")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Output("sp")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Output("partial")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Output("trace")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND});
    this->Output("stats").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Output("running").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    this->Attr("q_block").AttrType(REQUIRED).Int();
    this->Attr("k_block").AttrType(REQUIRED).Int();
    this->Attr("q_l1").AttrType(REQUIRED).Int();
    this->Attr("groups").AttrType(REQUIRED).Int();
    this->Attr("capture").AttrType(REQUIRED).Bool();
    this->Attr("phase").AttrType(REQUIRED).Int();
    this->SetInferShape(Infer).SetInferDataType(Types);
    this->AICore()
        .SetTiling(Tile)
        .AddConfig("ascend910b")
        .AddConfig("ascend910_93");
  }
};
OP_ADD(FaFaNkNzVectorFp32V5Sumpack);
} // namespace ops
