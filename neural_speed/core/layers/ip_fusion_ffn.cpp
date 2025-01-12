//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "bestla_common.hpp"

using namespace bestla;     // NOLINT
using namespace ne_bestla;  // NOLINT

unsigned long long bestla_fusion_FFN_f32f32_get_workspace_size(int seq, int fin, int fmid, int fout,  // NOLINT
                                                               void* w1ptr, void* w2ptr) {
  // lazy size: maximum padding
  int constexpr padding = 128;
  size_t s = static_cast<size_t>(seq) * utils::padto(static_cast<size_t>(fin), padding) * 4;
  s += static_cast<size_t>(seq) * utils::padto(static_cast<size_t>(fmid), padding) * 4;
  return s;
}

namespace ffn_2w {

template <class Parallel_T, class Launch_T1, class Launch_T2>
void GemmRunWithA_ffn(const typename Launch_T1::Param& args1, const typename Launch_T2::Param& args2,
                      parallel::IThreading* th) {
  parallel::gemm::SchedulerDispatcher<Parallel_T> para1({th, args1.problem});
  parallel::gemm::SchedulerDispatcher<Parallel_T> para2({th, args2.problem});
  using AParall1 = typename Launch_T1::PrologueA::Parallel;
  using AParall2 = typename Launch_T2::PrologueA::Parallel;
  auto apara1 = Launch_T1::PrologueA::createParallel(th->num_threads(), args1.problem);
  auto apara2 = Launch_T2::PrologueA::createParallel(th->num_threads(), args2.problem);
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para1.print();
    para2.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename AParall1::ThreadProblem thdpA1{tidx};
    apara1.getIndex(thdpA1);
    if (thdpA1.valid) {
      Launch_T1::PrologueA::run(args1.paramA, thdpA1);
    }
    th->sync(tidx, 0);
    typename Parallel_T::ThreadProblem thdp1{tidx};
    para1.getIndex(thdp1);
    if (thdp1.valid) {
      Launch_T1::run(args1, thdp1);
    }
    th->sync(tidx, 1);
    typename AParall2::ThreadProblem thdpA2{tidx};
    apara2.getIndex(thdpA2);
    if (thdpA2.valid) {
      Launch_T2::PrologueA::run(args2.paramA, thdpA2);
    }
    th->sync(tidx, 2);
    typename Parallel_T::ThreadProblem thdp2{tidx};
    para2.getIndex(thdp2);
    if (thdp2.valid) {
      Launch_T2::run(args2, thdp2);
    }
  });
}

template <class Parallel_T, class Launch_T1, class Launch_T2>
void GemmRun_ffn(const typename Launch_T1::Param& args1, const typename Launch_T2::Param& args2,
                 parallel::IThreading* th) {
  parallel::gemm::SchedulerDispatcher<Parallel_T> para1({th, args1.problem});
  parallel::gemm::SchedulerDispatcher<Parallel_T> para2({th, args2.problem});
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para1.print();
    para2.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename Parallel_T::ThreadProblem thdp1{tidx};
    para1.getIndex(thdp1);
    if (thdp1.valid) {
      Launch_T1::run(args1, thdp1);
    }
    th->sync(tidx);
    typename Parallel_T::ThreadProblem thdp2{tidx};
    para2.getIndex(thdp2);
    if (thdp2.valid) {
      Launch_T2::run(args2, thdp2);
    }
  });
}

template <class GemmCore_T, template <class> class Wei_T, template <class> class Act_T, class Epi_T1, class Epi_T2>
void BTLAGemmCompF32(float* activation, storage::gemm::IWeightBase* w1ptr, storage::gemm::IWeightBase* w2ptr,
                     float* tmp, float* output, int seq, int fin, int fmid, int fout, void* workspace,
                     parallel::IThreading* th, typename Epi_T1::Param epi_prama1, typename Epi_T2::Param epi_prama2) {
  using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
  using Launcher_epi = wrapper::gemm::LauncherBase<GemmCore_T, Act_T, Wei_T, Epi_T1>;
  using Launcher = wrapper::gemm::LauncherBase<GemmCore_T, Act_T, Wei_T, Epi_T2>;
  auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
  auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
  utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
  utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
  auto reordA1 = Launcher_epi::PrologueA::createReorderStorage(seq, fin, w1ptr_->mBlockSize);
  auto reordA2 = Launcher::PrologueA::createReorderStorage(seq, fin, w2ptr_->mBlockSize);
  typename Launcher_epi::Param args1{
      gp1, {activation, fin, nullptr, w1ptr_->ShfIndice(), &reordA1}, {w1ptr_}, epi_prama1};
  typename Launcher::Param args2{gp2, {tmp, fmid, nullptr, w2ptr_->ShfIndice(), &reordA2}, {w2ptr_}, epi_prama2};
  auto WS = reinterpret_cast<int8_t*>(workspace);
  if (w1ptr_->ShfIndice()) {
    reordA1.assign(WS);
    reordA2.assign(WS);
    GemmRunWithA_ffn<Parallel, Launcher_epi, Launcher>(args1, args2, th);
  } else {
    GemmRun_ffn<Parallel, Launcher_epi, Launcher>(args1, args2, th);
  }
}

template <class GemmCore_T, template <class> class Wei_T, class Epi_T1, class Epi_T2>
void BTLAGemmCompInt8Pc(float* activation, storage::gemm::IWeightBase* w1ptr, storage::gemm::IWeightBase* w2ptr,
                        float* tmp, float* output, int seq, int fin, int fmid, int fout, void* workspace,
                        parallel::IThreading* th, typename Epi_T1::Fp32Param epi_prama1,
                        typename Epi_T2::Fp32Param epi_prama2) {
  using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
  using Launcher_epi =
      wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T1>;
  using Launcher =
      wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T2>;
  auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
  auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
  utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
  utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
  assert(w1ptr_->mBlockSize >= fin);
  assert(w2ptr_->mBlockSize >= fmid);
  auto WS = reinterpret_cast<int8_t*>(workspace);
  auto quanA1 = Launcher_epi::PrologueA::createStorage(seq, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
  quanA1.assign(WS);
  WS += quanA1.mSize;
  auto reordA1 = Launcher_epi::PrologueA::createReorderStorage(seq, fin, w1ptr_->mBlockSize);
  if (w1ptr_->ShfIndice()) {
    reordA1.assign(WS);
  }
  WS = reinterpret_cast<int8_t*>(workspace);
  auto quanA2 = Launcher::PrologueA::createStorage(seq, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
  quanA2.assign(WS);
  WS += quanA2.mSize;
  auto reordA2 = Launcher::PrologueA::createReorderStorage(seq, fin, w2ptr_->mBlockSize);
  if (w2ptr_->ShfIndice()) {
    reordA2.assign(WS);
  }
  typename Launcher_epi::Param args1{
      gp1,
      {activation, fin, &quanA1, w1ptr_->ShfIndice(), &reordA1},
      {w1ptr_},
      {{w1ptr_->template SPtr<char>(), w1ptr_->SDtype(), quanA1.template SPtr<float>(), quanA1.template ZPtr<uint8_t>(),
        w1ptr_->template RPtr<char>(), w1ptr_->RDtype(), nullptr, nullptr, fin},
       epi_prama1}};
  typename Launcher::Param args2{
      gp2,
      {tmp, fmid, &quanA2, w2ptr_->ShfIndice(), &reordA2},
      {w2ptr_},
      {{w2ptr_->template SPtr<char>(), w2ptr_->SDtype(), quanA2.template SPtr<float>(), quanA2.template ZPtr<uint8_t>(),
        w2ptr_->template RPtr<char>(), w2ptr_->RDtype(), nullptr, nullptr, fmid},
       epi_prama2}};
  GemmRunWithA_ffn<Parallel, Launcher_epi, Launcher>(args1, args2, th);
}

template <class GemmCore_T, template <class> class Wei_T, class Epi_T1, class Epi_T2>
void BTLAGemmCompInt8(float* activation, storage::gemm::IWeightBase* w1ptr, storage::gemm::IWeightBase* w2ptr,
                      float* tmp, float* output, int seq, int fin, int fmid, int fout, void* workspace,
                      parallel::IThreading* th, typename Epi_T1::Param epi_prama1, typename Epi_T2::Param epi_prama2) {
  using Parallel = parallel::gemm::SchedulerKBlockS<GemmCore_T>;
  using Launcher_epi =
      wrapper::gemm::LauncherIntKBlock<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T1>;
  using Launcher =
      wrapper::gemm::LauncherIntKBlock<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T2>;
  auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
  auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
  utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
  utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
  auto WS = reinterpret_cast<int8_t*>(workspace);
  auto quanA1 = Launcher_epi::PrologueA::createStorage(seq, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
  quanA1.assign(WS);
  WS += quanA1.mSize;
  auto reordA1 = Launcher_epi::PrologueA::createReorderStorage(seq, fin, w1ptr_->mBlockSize);
  if (w1ptr_->ShfIndice()) {
    reordA1.assign(WS);
  }
  WS = reinterpret_cast<int8_t*>(workspace);
  auto quanA2 = Launcher::PrologueA::createStorage(seq, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
  quanA2.assign(WS);
  WS += quanA2.mSize;
  auto reordA2 = Launcher::PrologueA::createReorderStorage(seq, fin, w2ptr_->mBlockSize);
  if (w2ptr_->ShfIndice()) {
    reordA2.assign(WS);
  }
  typename Launcher_epi::Param args1{
      gp1, {activation, fin, &quanA1, w1ptr_->ShfIndice(), &reordA1}, {w1ptr_}, epi_prama1};
  typename Launcher::Param args2{gp2, {tmp, fmid, &quanA2, w2ptr_->ShfIndice(), &reordA2}, {w2ptr_}, epi_prama2};
  GemmRunWithA_ffn<Parallel, Launcher_epi, Launcher>(args1, args2, th);
}

bool bestla_fusion_ffn_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  GetCPUDevice();
  auto w1tmp = storage::gemm::PackedWeightParser::deserialBuffer(w1ptr);
  auto w2tmp = storage::gemm::PackedWeightParser::deserialBuffer(w2ptr);
  bool support = false;
  if (w1tmp != nullptr && w2tmp != nullptr) {
    auto sameKernel = samePackedWeight(w1tmp, w2tmp);
    if (sameKernel) {
      if (w1tmp->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
        constexpr size_t EleNum = sizeof(AllKBlockCores) / sizeof(AllKBlockCores[0]);
        support = contains(w1tmp->mCoreId, AllKBlockCores, EleNum);
        support &= hasISA(AllKBlockCores, EleNum);
      } else if (w1tmp->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNFloat) {
        constexpr size_t EleNum = sizeof(FloatCores) / sizeof(FloatCores[0]);
        support = contains(w1tmp->mCoreId, FloatCores, EleNum);
        support &= hasISA(FloatCores, EleNum);
      }
    }
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  return support;
}

template <class epilogue1, class epilogue2, typename Epi_args1, typename Epi_args2>
void bestla_fusion_ffn_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp, float* output, int seq,
                                      int fin, int fmid, int fout, void* workspace, Epi_args1 epi_args1,
                                      Epi_args2 epi_args2) {
  GetCPUDevice();
  auto pth = ne_threading::get();
  auto ptr1 = storage::gemm::PackedWeightParser::deserialBuffer(w1ptr);
  auto ptr2 = storage::gemm::PackedWeightParser::deserialBuffer(w2ptr);
  auto _workspace = reinterpret_cast<int8_t*>(workspace);
  if (ptr1) {
    auto coretype = ptr1->mCoreId;
    auto NTile = gemm::CoreAttr::get_mask_val(ptr1->mCoreId, gemm::CoreAttr::NTILE_MASK, gemm::CoreAttr::NTILE_SHIFT);
    auto PackRow = gemm::CoreAttr::get_packrow(ptr1->mCoreId);
    auto CType = gemm::CoreAttr::get_comp(ptr1->mCoreId);
    auto btype = static_cast<gemm::CompType>(gemm::CompTypeHelper::get_B(CType));
    if (ptr1->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto bptr = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(ptr1);
      auto bptr2 = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(ptr2);
      auto BlkSize = bptr->mBlockSize;
      auto BlkSize2 = bptr2->mBlockSize;
      if (btype == gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          BTLAGemmCompF32<tAVX512F, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          BTLAGemmCompF32<tAVX2, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        }
      }
      if (btype == gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          BTLAGemmCompF32<tAMX_BF16, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        }
      }
      if (btype == gemm::CompType::tFP16 && PackRow == 2) {
        if (NTile == tAMX_FP16::NTILE && _cd->AMX_FP16() && BlkSize % tAMX_FP16::KTILE == 0) {
          BTLAGemmCompF32<tAMX_FP16, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        }
      }
      if (btype == gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_US_KBlock::NTILE && _cd->AMX_INT8() && BlkSize % tAMX_INT8_US_KBlock::KTILE == 0) {
          if (BlkSize < fin || BlkSize2 < fmid) {
            BTLAGemmCompInt8<tAMX_INT8_US_KBlock, tWeiNInt, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          } else {
            BTLAGemmCompInt8Pc<tAMX_INT8_US, tWeiNInt, epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue1>,
                               epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue2>>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          }

        } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI() &&
                   BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          if (BlkSize < fin || BlkSize2 < fmid) {
            BTLAGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          } else {
            BTLAGemmCompInt8Pc<tAVX512_VNNI, tWeiNInt, epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue1>,
                               epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue2>>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          }
        } else if (NTile == tAVX512BW_KBlock::NTILE && _cd->AVX512BW() && BlkSize % tAVX512BW_KBlock::KTILE == 0) {
          if (BlkSize < fin || BlkSize2 < fmid) {
            BTLAGemmCompInt8<tAVX512BW_KBlock, tWeiNInt, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          } else {
            BTLAGemmCompInt8Pc<tAVX512BW, tWeiNInt, epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue1>,
                               epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue2>>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          }
        } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          if (BlkSize < fin || BlkSize2 < fmid) {
            BTLAGemmCompInt8<tAVX_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          } else {
            BTLAGemmCompInt8Pc<tAVX_VNNI, tWeiNInt, epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue1>,
                               epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue2>>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          }
        } else if (NTile == tAVX2_VNNI_KBlock::NTILE && _cd->AVX2() && BlkSize % tAVX2_VNNI_KBlock::KTILE == 0) {
          if (BlkSize < fin || BlkSize2 < fmid) {
            BTLAGemmCompInt8<tAVX2_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          } else {
            BTLAGemmCompInt8Pc<tAVX2_VNNI, tWeiNInt, epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue1>,
                               epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue2>>(
                activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
          }
        }
      }
    }
    if (ptr1->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNFloat) {
      auto bptr = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(ptr1);
      auto BlkSize = bptr->mBlockSize;
      if (btype == gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          BTLAGemmCompF32<tAVX512F, tWeiNFloat, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          BTLAGemmCompF32<tAVX2, tWeiNFloat, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        }
      }
      if (btype == gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          BTLAGemmCompF32<tAMX_BF16, tWeiNFloat, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        }
      }
      if (btype == gemm::CompType::tFP16 && PackRow == 2) {
        if (NTile == tAMX_FP16::NTILE && _cd->AMX_FP16() && BlkSize % tAMX_FP16::KTILE == 0) {
          BTLAGemmCompF32<tAMX_FP16, tWeiNFloat, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, tmp, output, seq, fin, fmid, fout, workspace, pth, epi_args1, epi_args2);
        }
      }
    }
    delete ptr1;
    delete ptr2;
  } else {
    printf("Wrong Input\n");
    assert(0);
  }
}
}  // namespace ffn_2w

namespace ffn_3w {

template <class Parallel_T, class Launch_T1, class Launch_T2, class Launch_T3>
void GemmRunWithA_ffn(const typename Launch_T1::Param& args1, const typename Launch_T2::Param& args2,
                      const typename Launch_T3::Param& args3, parallel::IThreading* th) {
  parallel::gemm::SchedulerDispatcher<Parallel_T> para1({th, args1.problem});
  parallel::gemm::SchedulerDispatcher<Parallel_T> para3({th, args3.problem});
  using AParall1 = typename Launch_T1::PrologueA::Parallel;
  using AParall3 = typename Launch_T3::PrologueA::Parallel;
  auto apara1 = Launch_T1::PrologueA::createParallel(th->num_threads(), args1.problem);
  auto apara3 = Launch_T3::PrologueA::createParallel(th->num_threads(), args3.problem);
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para1.print();
    para3.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename AParall1::ThreadProblem thdpA1{tidx};
    apara1.getIndex(thdpA1);
    if (thdpA1.valid) {
      Launch_T1::PrologueA::run(args1.paramA, thdpA1);
    }
    th->sync(tidx, 0);
    typename Parallel_T::ThreadProblem thdp1{tidx};
    para1.getIndex(thdp1);
    if (thdp1.valid) {
      Launch_T1::run(args1, thdp1);
      Launch_T2::run(args2, thdp1);
    }
    th->sync(tidx, 1);
    typename AParall3::ThreadProblem thdpA3{tidx};
    apara3.getIndex(thdpA3);
    if (thdpA3.valid) {
      Launch_T3::PrologueA::run(args3.paramA, thdpA3);
    }
    th->sync(tidx, 2);
    typename Parallel_T::ThreadProblem thdp3{tidx};
    para3.getIndex(thdp3);
    if (thdp3.valid) {
      Launch_T3::run(args3, thdp3);
    }
  });
}

template <class Parallel_T, class Launch_T1, class Launch_T2, class Launch_T3>
void GemmRun_ffn(const typename Launch_T1::Param& args1, const typename Launch_T2::Param& args2,
                 const typename Launch_T3::Param& args3, parallel::IThreading* th) {
  parallel::gemm::SchedulerDispatcher<Parallel_T> para1({th, args1.problem});
  parallel::gemm::SchedulerDispatcher<Parallel_T> para3({th, args3.problem});
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para1.print();
    para3.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename Parallel_T::ThreadProblem thdp1{tidx};
    para1.getIndex(thdp1);
    if (thdp1.valid) {
      Launch_T1::run(args1, thdp1);
      Launch_T2::run(args2, thdp1);
    }
    th->sync(tidx);
    typename Parallel_T::ThreadProblem thdp3{tidx};
    para3.getIndex(thdp3);
    if (thdp3.valid) {
      Launch_T3::run(args3, thdp3);
    }
  });
}

template <class GemmCore_T, template <class> class Wei_T, template <class> class Act_T, class Epi_T1, class Epi_T2>
void BTLAGemmCompF32(float* activation, storage::gemm::IWeightBase* w1ptr, storage::gemm::IWeightBase* w2ptr,
                     storage::gemm::IWeightBase* w3ptr, float* tmp1, float* tmp2, float* output, int seq, int fin,
                     int fmid, int fout, void* workspace, parallel::IThreading* th, typename Epi_T1::Param epi_prama1,
                     typename Epi_T2::Param epi_prama2) {
  using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
  using Launcher_epi = wrapper::gemm::LauncherBase<GemmCore_T, Act_T, Wei_T, Epi_T1>;
  using Launcher_mul = wrapper::gemm::LauncherBase<GemmCore_T, Act_T, Wei_T, custom::epilogue::MulFp32>;
  using Launcher = wrapper::gemm::LauncherBase<GemmCore_T, Act_T, Wei_T, Epi_T2>;
  auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
  auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
  auto w3ptr_ = reinterpret_cast<typename Launcher_mul::PrologueB::StorageWeight*>(w3ptr);
  assert(w1ptr_->ShfIndice() == nullptr);
  assert(w2ptr_->ShfIndice() == nullptr);
  assert(w3ptr_->ShfIndice() == nullptr);
  utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
  utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
  utils::GemmProblem gp3(1, seq, fmid, fin, w3ptr_->mBlockSize);
  typename Launcher_epi::Param args1{gp1, {activation, fin, nullptr}, {w1ptr_}, epi_prama1};
  typename Launcher::Param args2{gp2, {tmp2, fmid, nullptr}, {w2ptr_}, epi_prama2};
  typename Launcher_mul::Param args3{gp3, {activation, fin, nullptr}, {w3ptr_}, {tmp2, tmp1, fmid, fmid}};
  GemmRun_ffn<Parallel, Launcher_epi, Launcher_mul, Launcher>(args1, args3, args2, th);
}

template <class GemmCore_T, template <class> class Wei_T, class Epi_T1, class Epi_T2>
void BTLAGemmCompInt8Pc(float* activation, storage::gemm::IWeightBase* w1ptr, storage::gemm::IWeightBase* w2ptr,
                        storage::gemm::IWeightBase* w3ptr, float* tmp1, float* tmp2, float* output, int seq, int fin,
                        int fmid, int fout, void* workspace, parallel::IThreading* th,
                        typename Epi_T1::Fp32Param epi_prama1, typename Epi_T2::Fp32Param epi_prama2) {
  using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
  using Launcher_epi =
      wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T1>;
  using Launcher_mul =
      wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T,
                                  epilogue::gemm::PcKBlockCompInt8Epilogue<custom::epilogue::MulFp32>>;
  using Launcher =
      wrapper::gemm::LauncherBase<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T2>;
  auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
  auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
  auto w3ptr_ = reinterpret_cast<typename Launcher_mul::PrologueB::StorageWeight*>(w3ptr);
  utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
  utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
  utils::GemmProblem gp3(1, seq, fmid, fin, w3ptr_->mBlockSize);
  auto quanA1 = Launcher_epi::PrologueA::createStorage(seq, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
  auto WS = reinterpret_cast<int8_t*>(workspace);
  quanA1.assign(WS);

  auto quanA2 = Launcher::PrologueA::createStorage(seq, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
  WS = reinterpret_cast<int8_t*>(workspace);
  quanA2.assign(WS);
  assert(w1ptr_->ShfIndice() == nullptr);
  assert(w2ptr_->ShfIndice() == nullptr);
  assert(w3ptr_->ShfIndice() == nullptr);
  typename Launcher_epi::Param args1{
      gp1,
      {activation, fin, &quanA1},
      {w1ptr_},
      {{w1ptr_->template SPtr<char>(), w1ptr_->SDtype(), quanA1.template SPtr<float>(), quanA1.template ZPtr<uint8_t>(),
        w1ptr_->template RPtr<char>(), w1ptr_->RDtype(), nullptr, nullptr, fin},
       epi_prama1}};
  typename Launcher::Param args2{
      gp2,
      {tmp2, fmid, &quanA2},
      {w2ptr_},
      {{w2ptr_->template SPtr<char>(), w2ptr_->SDtype(), quanA2.template SPtr<float>(), quanA2.template ZPtr<uint8_t>(),
        w2ptr_->template RPtr<char>(), w2ptr_->RDtype(), nullptr, nullptr, fmid},
       epi_prama2}};
  typename Launcher_mul::Param args3{
      gp3,
      {activation, fin, &quanA1},
      {w3ptr_},
      {{w3ptr_->template SPtr<char>(), w3ptr_->SDtype(), quanA1.template SPtr<float>(), quanA1.template ZPtr<uint8_t>(),
        w3ptr_->template RPtr<char>(), w3ptr_->RDtype(), nullptr, nullptr, fin},
       {tmp2, tmp1, fmid, fmid}}};
  GemmRunWithA_ffn<Parallel, Launcher_epi, Launcher_mul, Launcher>(args1, args3, args2, th);
}

template <class GemmCore_T, template <class> class Wei_T, class Epi_T1, class Epi_T2>
void BTLAGemmCompInt8(float* activation, storage::gemm::IWeightBase* w1ptr, storage::gemm::IWeightBase* w2ptr,
                      storage::gemm::IWeightBase* w3ptr, float* tmp1, float* tmp2, float* output, int seq, int fin,
                      int fmid, int fout, void* workspace, parallel::IThreading* th, typename Epi_T1::Param epi_prama1,
                      typename Epi_T2::Param epi_prama2) {
  using Parallel = parallel::gemm::SchedulerKBlockS<GemmCore_T>;
  using Launcher_epi =
      wrapper::gemm::LauncherIntKBlock<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T1>;
  using Launcher_mul =
      wrapper::gemm::LauncherIntKBlock<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T,
                                       custom::epilogue::MulFp32>;
  using Launcher =
      wrapper::gemm::LauncherIntKBlock<GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32, Wei_T, Epi_T2>;
  auto w1ptr_ = reinterpret_cast<typename Launcher_epi::PrologueB::StorageWeight*>(w1ptr);
  auto w2ptr_ = reinterpret_cast<typename Launcher::PrologueB::StorageWeight*>(w2ptr);
  auto w3ptr_ = reinterpret_cast<typename Launcher_mul::PrologueB::StorageWeight*>(w3ptr);
  utils::GemmProblem gp1(1, seq, fmid, fin, w1ptr_->mBlockSize);
  utils::GemmProblem gp2(1, seq, fout, fmid, w2ptr_->mBlockSize);
  utils::GemmProblem gp3(1, seq, fmid, fin, w3ptr_->mBlockSize);
  auto quanA1 = Launcher_epi::PrologueA::createStorage(seq, fin, w1ptr_->mBlockSize, w1ptr_->IsAsym());
  auto WS = reinterpret_cast<int8_t*>(workspace);
  quanA1.assign(WS);

  auto quanA2 = Launcher::PrologueA::createStorage(seq, fmid, w2ptr_->mBlockSize, w2ptr_->IsAsym());
  WS = reinterpret_cast<int8_t*>(workspace);
  quanA2.assign(WS);
  assert(w1ptr_->ShfIndice() == nullptr);
  assert(w2ptr_->ShfIndice() == nullptr);
  assert(w3ptr_->ShfIndice() == nullptr);
  typename Launcher_epi::Param args1{gp1, {activation, fin, &quanA1}, {w1ptr_}, epi_prama1};
  typename Launcher::Param args2{gp2, {tmp2, fmid, &quanA2}, {w2ptr_}, epi_prama2};
  typename Launcher_mul::Param args3{gp3, {activation, fin, &quanA1}, {w3ptr_}, {tmp2, tmp1, fmid, fmid}};
  GemmRunWithA_ffn<Parallel, Launcher_epi, Launcher_mul, Launcher>(args1, args3, args2, th);
}

bool bestla_fusion_ffn_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid, int fout) {
  GetCPUDevice();
  auto w1tmp = storage::gemm::PackedWeightParser::deserialBuffer(w1ptr);
  auto w2tmp = storage::gemm::PackedWeightParser::deserialBuffer(w2ptr);
  auto w3tmp = storage::gemm::PackedWeightParser::deserialBuffer(w3ptr);
  bool support = false;
  if (w1tmp != nullptr && w2tmp != nullptr && w3tmp != nullptr) {
    storage::gemm::IWeightBase* tmps[3] = {w1tmp, w2tmp, w3tmp};
    auto sameKernel = samePackedWeight(tmps, 3);
    if (sameKernel) {
      if (w1tmp->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
        auto w1ptr = reinterpret_cast<storage::gemm::StorageWeightKBlockNInteger*>(w1tmp);
        if (w1ptr->ShfIndice()) {
          return false;  // Do not support 3w ffn fusion for activation shuffle
        }
        constexpr size_t EleNum = sizeof(AllKBlockCores) / sizeof(AllKBlockCores[0]);
        support = contains(w1tmp->mCoreId, AllKBlockCores, EleNum);
        support &= hasISA(AllKBlockCores, EleNum);
      } else if (w1tmp->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNFloat) {
        constexpr size_t EleNum = sizeof(FloatCores) / sizeof(FloatCores[0]);
        support = contains(w1tmp->mCoreId, FloatCores, EleNum);
        support &= hasISA(FloatCores, EleNum);
      }
    }
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  safe_delete(w3tmp);
  return support;
}

template <class epilogue1, class epilogue2, typename Epi_args1, typename Epi_args2>
void bestla_fusion_ffn_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                      float* tmp2, float* output, int seq, int fin, int fmid, int fout, void* workspace,
                                      Epi_args1 epi_args1, Epi_args2 epi_args2) {
  GetCPUDevice();
  auto pth = ne_threading::get();
  auto ptr1 = storage::gemm::PackedWeightParser::deserialBuffer(w1ptr);
  auto ptr2 = storage::gemm::PackedWeightParser::deserialBuffer(w2ptr);
  auto ptr3 = storage::gemm::PackedWeightParser::deserialBuffer(w3ptr);
  auto _workspace = reinterpret_cast<int8_t*>(workspace);
  if (ptr1) {
    auto coretype = ptr1->mCoreId;
    auto NTile = gemm::CoreAttr::get_mask_val(ptr1->mCoreId, gemm::CoreAttr::NTILE_MASK, gemm::CoreAttr::NTILE_SHIFT);
    auto PackRow = gemm::CoreAttr::get_packrow(ptr1->mCoreId);
    auto CType = gemm::CoreAttr::get_comp(ptr1->mCoreId);
    auto btype = static_cast<gemm::CompType>(gemm::CompTypeHelper::get_B(CType));
    if (ptr1->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto bptr = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(ptr1);
      auto bptr2 = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(ptr2);
      auto BlkSize = bptr->mBlockSize;
      auto BlkSize2 = bptr2->mBlockSize;
      if (btype == gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          BTLAGemmCompF32<tAVX512F, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1,
                                                                                  tmp2, output, seq, fin, fmid, fout,
                                                                                  workspace, pth, epi_args1, epi_args2);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          BTLAGemmCompF32<tAVX2, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1, tmp2,
                                                                               output, seq, fin, fmid, fout, workspace,
                                                                               pth, epi_args1, epi_args2);
        }
      }
      if (btype == gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          BTLAGemmCompF32<tAMX_BF16, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
              epi_args2);
        }
      }
      if (btype == gemm::CompType::tFP16 && PackRow == 2) {
        if (NTile == tAMX_FP16::NTILE && _cd->AMX_FP16() && BlkSize % tAMX_FP16::KTILE == 0) {
          BTLAGemmCompF32<tAMX_FP16, tWeiNInt, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
              epi_args2);
        }
      }
      if (btype == gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_US_KBlock::NTILE && _cd->AMX_INT8() && BlkSize % tAMX_INT8_US_KBlock::KTILE == 0) {
          if (BlkSize < fin || BlkSize2 < fmid) {
            BTLAGemmCompInt8<tAMX_INT8_US_KBlock, tWeiNInt, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1,
                                                                                  tmp2, output, seq, fin, fmid, fout,
                                                                                  workspace, pth, epi_args1, epi_args2);
          } else {
            BTLAGemmCompInt8Pc<tAMX_INT8_US, tWeiNInt, epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue1>,
                               epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue2>>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          }

        } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI() &&
                   BlkSize % tAVX512_VNNI_KBlock::KTILE == 0) {
          if (BlkSize < fin || BlkSize2 < fmid) {
            BTLAGemmCompInt8<tAVX512_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1,
                                                                                  tmp2, output, seq, fin, fmid, fout,
                                                                                  workspace, pth, epi_args1, epi_args2);
          } else {
            BTLAGemmCompInt8Pc<tAVX512_VNNI, tWeiNInt, epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue1>,
                               epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue2>>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          }
        } else if (NTile == tAVX512BW_KBlock::NTILE && _cd->AVX512BW() && BlkSize % tAVX512BW_KBlock::KTILE == 0) {
          if (BlkSize < fin || BlkSize2 < fmid) {
            BTLAGemmCompInt8<tAVX512BW_KBlock, tWeiNInt, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1, tmp2,
                                                                               output, seq, fin, fmid, fout, workspace,
                                                                               pth, epi_args1, epi_args2);
          } else {
            BTLAGemmCompInt8Pc<tAVX512BW, tWeiNInt, epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue1>,
                               epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue2>>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          }
        } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI() && BlkSize % tAVX_VNNI_KBlock::KTILE == 0) {
          if (BlkSize < fin || BlkSize2 < fmid) {
            BTLAGemmCompInt8<tAVX_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1, tmp2,
                                                                               output, seq, fin, fmid, fout, workspace,
                                                                               pth, epi_args1, epi_args2);
          } else {
            BTLAGemmCompInt8Pc<tAVX_VNNI, tWeiNInt, epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue1>,
                               epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue2>>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          }
        } else if (NTile == tAVX2_VNNI_KBlock::NTILE && _cd->AVX2() && BlkSize % tAVX2_VNNI_KBlock::KTILE == 0) {
          if (BlkSize < fin || BlkSize2 < fmid) {
            BTLAGemmCompInt8<tAVX2_VNNI_KBlock, tWeiNInt, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1,
                                                                                tmp2, output, seq, fin, fmid, fout,
                                                                                workspace, pth, epi_args1, epi_args2);
          } else {
            BTLAGemmCompInt8Pc<tAVX2_VNNI, tWeiNInt, epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue1>,
                               epilogue::gemm::PcKBlockCompInt8Epilogue<epilogue2>>(
                activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
                epi_args2);
          }
        }
      }
    }
    if (ptr1->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNFloat) {
      auto bptr = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(ptr1);
      auto BlkSize = bptr->mBlockSize;
      if (btype == gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE && _cd->AVX512F() && BlkSize % tAVX512F::KTILE == 0) {
          BTLAGemmCompF32<tAVX512F, tWeiNFloat, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
              epi_args2);
        } else if (NTile == tAVX2::NTILE && _cd->AVX2() && BlkSize % tAVX2::KTILE == 0) {
          BTLAGemmCompF32<tAVX2, tWeiNFloat, tActKBaseF32, epilogue1, epilogue2>(activation, ptr1, ptr2, ptr3, tmp1,
                                                                                 tmp2, output, seq, fin, fmid, fout,
                                                                                 workspace, pth, epi_args1, epi_args2);
        }
      }
      if (btype == gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16() && BlkSize % tAMX_BF16::KTILE == 0) {
          BTLAGemmCompF32<tAMX_BF16, tWeiNFloat, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
              epi_args2);
        }
      }
      if (btype == gemm::CompType::tFP16 && PackRow == 2) {
        if (NTile == tAMX_FP16::NTILE && _cd->AMX_FP16() && BlkSize % tAMX_FP16::KTILE == 0) {
          BTLAGemmCompF32<tAMX_FP16, tWeiNFloat, tActKBaseF32, epilogue1, epilogue2>(
              activation, ptr1, ptr2, ptr3, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, pth, epi_args1,
              epi_args2);
        }
      }
    }
    delete ptr1;
    delete ptr2;
    delete ptr3;
  } else {
    printf("Wrong Input\n");
    assert(0);
  }
}
}  // namespace ffn_3w

bool bestla_fusion_FFN_SiLu_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid,
                                           int fout) {
  return ffn_3w::bestla_fusion_ffn_f32f32_support(w1ptr, w2ptr, w3ptr, seq, fin, fmid, fout);
}

bool bestla_fusion_FFN_Gelu_Mul_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid,
                                               int fout) {
  return ffn_3w::bestla_fusion_ffn_f32f32_support(w1ptr, w2ptr, w3ptr, seq, fin, fmid, fout);
}

void bestla_fusion_FFN_SiLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                           float* tmp2, float* output, int seq, int fin, int fmid, int fout,
                                           void* workspace) {
  float silu_alpha = -1.0f;
  epilogue::gemm::ParamAccumulatorWriteBack<float> epi_args1 = {tmp1, fmid, &silu_alpha};
  epilogue::gemm::ParamAccumulatorWriteBack<float> epi_args2 = {output, fout};
  ffn_3w::bestla_fusion_ffn_f32f32_forward<epilogue::gemm::AccumulatorWriteBackWithSwishFp32,
                                           epilogue::gemm::AccumulatorWriteBackFp32>(
      activation, w1ptr, w2ptr, w3ptr, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, epi_args1, epi_args2);
}

void bestla_fusion_FFN_Gelu_Mul_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                               float* tmp2, float* output, int seq, int fin, int fmid, int fout,
                                               void* workspace) {
  epilogue::gemm::ParamAccumulatorWriteBack<float> epi_args1 = {tmp1, fmid};
  epilogue::gemm::ParamAccumulatorWriteBack<float> epi_args2 = {output, fout};
  ffn_3w::bestla_fusion_ffn_f32f32_forward<epilogue::gemm::AccumulatorWriteBackWithGeluFp32,
                                           epilogue::gemm::AccumulatorWriteBackFp32>(
      activation, w1ptr, w2ptr, w3ptr, tmp1, tmp2, output, seq, fin, fmid, fout, workspace, epi_args1, epi_args2);
}

bool bestla_fusion_FFN_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  return ffn_2w::bestla_fusion_ffn_f32f32_support(w1ptr, w2ptr, seq, fin, fmid, fout);
}

void bestla_fusion_FFN_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp1, float* output,
                                           int seq, int fin, int fmid, int fout, void* workspace) {
  epilogue::gemm::ParamAccumulatorWriteBack<float> epi_args1 = {tmp1, fmid, nullptr};
  epilogue::gemm::ParamAccumulatorWriteBack<float> epi_args2 = {output, fout};
  ffn_2w::bestla_fusion_ffn_f32f32_forward<epilogue::gemm::AccumulatorWriteBackWithGeluFp32,
                                           epilogue::gemm::AccumulatorWriteBackFp32>(
      activation, w1ptr, w2ptr, tmp1, output, seq, fin, fmid, fout, workspace, epi_args1, epi_args2);
}

bool bestla_fusion_FFN_Add_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  return ffn_2w::bestla_fusion_ffn_f32f32_support(w1ptr, w2ptr, seq, fin, fmid, fout);
}

void bestla_fusion_FFN_Add_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* b1ptr, float* b2ptr,
                                               float* tmp1, float* output, int seq, int fin, int fmid, int fout,
                                               bool broadcast_bias, void* workspace) {
  custom::epilogue::ParamAdd_Gelu<float> epi_args1 = {tmp1, b1ptr, fmid, broadcast_bias ? 0 : fmid};
  custom::epilogue::ParamAdd<float> epi_args2 = {output, b2ptr, fout, broadcast_bias ? 0 : fout};
  ffn_2w::bestla_fusion_ffn_f32f32_forward<custom::epilogue::Add_GeluFp32, custom::epilogue::AddFp32>(
      activation, w1ptr, w2ptr, tmp1, output, seq, fin, fmid, fout, workspace, epi_args1, epi_args2);
}
