//===- AArch64RegBankSelect.cpp --------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the AArch64RegBankSelect class.
//===----------------------------------------------------------------------===//

// #include "AArch64RegisterBankInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include <array>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "aarch64-regbank-select"

// FIXME: remove
namespace llvm {
void initializeAArch64RegBankSelectPass(PassRegistry &Registry);
}

namespace {

/// The AArch64 register bank kinds. Uncertainty is modelled with Or.
enum RegisterBankKind {
  /// R0 - R15
  GPR,
  /// D0 - D31
  FPR,
  /// R or D registers
  GPROrFPR,
  /// Z0 - Z31
  SVEData,
  /// P0 - P15
  SVEPredicate,
  /// ZA*
  SME
};

/// Algorithm
///
/// 1. Categorize llvm::MachineInstr in #RegisterBankKind as complete and
/// precise as possible. Uncertainty is modelled with Or.
/// 2. Assign register banks for unambiguous llvm::MachineInstr.
/// 3. Use assigned register banks to assign ambiguous llvm::MachineInstr to
/// register banks.
///
/// Claim: In the third step, there are only a few MIs and a lot of assigned
/// register banks.
///
/// No recursion.
///
/// The most important methods are
/// AArch64RegBankSelect::assignAmbiguousRegisterBank,
/// AArch64RegBankSelect::classifyDef and
/// AArch64RegBankSelect::classifyMemoryDef.
///
/// GenericOpcodes.td on August 14 2023
/// Please extend.
///
/// Some Opcodes are ignored, e.g., G_JUMP_TABLE, G_STACKSAVE,
/// G_STACKRESTORE, and ...
/// Libcall: G_GET_FPMODE, G_SET_FPMODE, and G_RESET_FPMODE
///
class AArch64RegBankSelect : public llvm::RegBankSelect {
  /// classifiers
  RegisterBankKind classifyDef(const llvm::MachineInstr &) const;
  /// Generic classifier. It does not know the G_ .
  RegisterBankKind getDefRegisterBank(const llvm::MachineInstr &MI) const;
  /// Classify G_LOAD and G_STORE
  RegisterBankKind classifyMemoryDef(const llvm::MachineInstr &MI) const;
  /// Classify G_ATOMIC*
  RegisterBankKind classifyAtomicDef(const llvm::MachineInstr &MI) const;
  /// Classify G_INTRINSIC_*
  RegisterBankKind classifyIntrinsicDef(const llvm::MachineInstr &MI) const;
  /// Classify G_EXTRACT.
  RegisterBankKind classifyExtract(const llvm::MachineInstr &MI) const;
  /// Classify G_BUILD_VECTOR.
  RegisterBankKind classifyBuildVector(const llvm::MachineInstr &MI) const;
  /// Classify COPY.
  RegisterBankKind classifyCopy(const llvm::MachineInstr &MI) const;

  /// predicates
  bool isDomainReassignable(const llvm::MachineInstr &) const;
  bool usesFPR(const llvm::MachineInstr &) const;
  bool defsFPR(const llvm::MachineInstr &) const;
  bool usesGPR(const llvm::MachineInstr &) const;
  bool defsGPR(const llvm::MachineInstr &) const;
  bool isUnambiguous(const llvm::MachineInstr &) const;
  bool isAmbiguous(const llvm::MachineInstr &) const;
  bool isFloatingPoint(const llvm::MachineInstr &) const;
  bool isFloatingPointIntrinsic(const llvm::MachineInstr &) const;
  bool usesFPRRegisterBank(const llvm::MachineInstr &MI) const;
  bool usesGPRRegisterBank(const llvm::MachineInstr &MI) const;
  bool isUnassignable(const MachineInstr &MI) const;

  /// assign MIs to register banks
  void assignAmbiguousRegisterBank(const llvm::MachineInstr &);
  void assignUnambiguousRegisterBank(const llvm::MachineInstr &);
  void assignFPR(const llvm::MachineInstr &MI);
  void assignGPR(const llvm::MachineInstr &MI);

public:
  static char ID;

  AArch64RegBankSelect();

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  llvm::StringRef getPassName() const override {
    return "AArch64RegBankSelect";
  }

  llvm::MachineFunctionProperties getRequiredProperties() const override {
    return llvm::MachineFunctionProperties()
        .set(llvm::MachineFunctionProperties::Property::IsSSA)
        .set(llvm::MachineFunctionProperties::Property::Legalized);
  }

private:
  /// Current optimization remark emitter. Used to report failures.
  std::unique_ptr<llvm::MachineOptimizationRemarkEmitter> MORE;
};

struct UseDefBehavior {
  unsigned Opcode;
  bool uses;
  bool defs;
};

} // namespace

// FIXME: sort?

/// List of floating point instructions with their unconditional use/def
/// behavior.
static constexpr std::array<UseDefBehavior, 20> FloatingPoint = {
    {TargetOpcode::G_FCMP, true, true}, // maybe
    {TargetOpcode::G_LROUND, true, false},
    {TargetOpcode::G_LLROUND, true, false},
    {TargetOpcode::G_STRICT_FADD, true, true},
    {TargetOpcode::G_STRICT_FSUB, true, true},
    {TargetOpcode::G_STRICT_FMUL, true, true},
    {TargetOpcode::G_STRICT_FDIV, true, true},
    {TargetOpcode::G_STRICT_FREM, true, true},
    {TargetOpcode::G_STRICT_FMA, true, true},
    {TargetOpcode::G_STRICT_FSQRT, true, true},
    {TargetOpcode::G_STRICT_FLDEXP, true, true},
    {TargetOpcode::G_FPTOSI, true, false},
    {TargetOpcode::G_FPTOUI, true, false},
    {TargetOpcode::G_SITOFP, false, true},
    {TargetOpcode::G_UITOFP, false, true},
    {TargetOpcode::G_FPEXT, true, true},
    {TargetOpcode::G_FPTRUNC, true, true},
    {TargetOpcode::G_FABS, true, true},
    {TargetOpcode::G_FCOPYSIGN, true, true},
    {TargetOpcode::G_FCANONICALIZE, true, true},
    {TargetOpcode::G_IS_FPCLASS, true, true},
    {TargetOpcode::G_FMINNUM, true, true},
    {TargetOpcode::G_FMAXNUM, true, true},
    {TargetOpcode::G_FMINNUM_IEEE, true, true},
    {TargetOpcode::G_FMAXNUM_IEEE, true, true},
    {TargetOpcode::G_FMINIMUM, true, true},
    {TargetOpcode::G_FMAXIMUM, true, true},
    {TargetOpcode::G_FADD, true, true},
    {TargetOpcode::G_FSUB, true, true},
    {TargetOpcode::G_FMUL, true, true},
    {TargetOpcode::G_FMA, true, true},
    {TargetOpcode::G_FMAD, true, true},
    {TargetOpcode::G_FDIV, true, true},
    {TargetOpcode::G_FREM, true, true},
    {TargetOpcode::G_FPOW, true, true},
    {TargetOpcode::G_FPOWI, true, true},
    {TargetOpcode::G_FEXP, true, true},
    {TargetOpcode::G_FEXP2, true, true},
    {TargetOpcode::G_FEXP10, true, true},
    {TargetOpcode::G_FLOG, true, true},
    {TargetOpcode::G_FLOG2, true, true},
    {TargetOpcode::G_FLOG10, true, true},
    {TargetOpcode::G_FLDEXP, true, true},
    {TargetOpcode::G_FFREXP, true, true},
    {TargetOpcode::G_FCEIL, true, true},
    {TargetOpcode::G_FCOS, true, true},
    {TargetOpcode::G_FSIN, true, true},
    {TargetOpcode::G_FSQRT, true, true},
    {TargetOpcode::G_FFLOOR, true, true},
    {TargetOpcode::G_FRINT, true, true},
    {TargetOpcode::G_FNEARBYINT, true, true},
    {TargetOpcode::G_VECREDUCE_SEQ_FADD, true, true},
    {TargetOpcode::G_VECREDUCE_SEQ_FMUL, true, true},
    {TargetOpcode::G_VECREDUCE_FADD, true, true},
    {TargetOpcode::G_VECREDUCE_FMUL, true, true},
    {TargetOpcode::G_VECREDUCE_FMAX, true, true},
    {TargetOpcode::G_VECREDUCE_FMIN, true, true},
    {TargetOpcode::G_VECREDUCE_ADD, true, true},
    {TargetOpcode::G_VECREDUCE_MUL, true, true},
    {TargetOpcode::G_VECREDUCE_AND, true, true},
    {TargetOpcode::G_VECREDUCE_OR, true, true},
    {TargetOpcode::G_VECREDUCE_XOR, true, true},
    {TargetOpcode::G_VECREDUCE_SMAX, true, true},
    {TargetOpcode::G_VECREDUCE_SMIN, true, true},
    {TargetOpcode::G_VECREDUCE_UMAX, true, true},
    {TargetOpcode::G_VECREDUCE_FMAXIMUM, true, true},
    {TargetOpcode::G_VECREDUCE_FMINIMUM, true, true},
    {TargetOpcode::G_VECREDUCE_UMIN, true, true},
    {TargetOpcode::G_FCONSTANT, false, true},
    {TargetOpcode::G_INTRINSIC_TRUNC, true, true},
    {TargetOpcode::G_INTRINSIC_ROUND, true, true},
    {TargetOpcode::G_INTRINSIC_ROUNDEVEN, true, true},
    {AArch64::G_DUP, true, true}};

/// List of integer/GPR instructions with their unconditional use/def behavior.
static constexpr UseDefBehavior Integer[] = {
    {TargetOpcode::G_PTRMASK, true, true},
    {TargetOpcode::G_SDIV, true, true},
    {TargetOpcode::G_UDIV, true, true},
    {TargetOpcode::G_SMULO, true, true},
    {TargetOpcode::G_UMULO, true, true},
    {TargetOpcode::G_SADDE, true, true},
    {TargetOpcode::G_SSUBE, true, true},
    {TargetOpcode::G_UADDE, true, true},
    {TargetOpcode::G_USUBE, true, true},
    {TargetOpcode::G_SADDO, true, true},
    {TargetOpcode::G_SSUBO, true, true},
    {TargetOpcode::G_UADDO, true, true},
    {TargetOpcode::G_USUBO, true, true},
    {TargetOpcode::G_FPTOSI, false, true},
    {TargetOpcode::G_FPTOUI, false, true},
    {TargetOpcode::G_SITOFP, true, false},
    {TargetOpcode::G_UITOFP, true, false},
    {TargetOpcode::G_CONSTANT, true, true},
    {TargetOpcode::G_SEXT_INREG, true, true},
    {TargetOpcode::G_BRCOND, true, true},
    {TargetOpcode::G_ATOMIC_CMPXCHG_WITH_SUCCESS, true, true},
    {TargetOpcode::G_ATOMIC_CMPXCHG, true, true},
    {TargetOpcode::G_ATOMICRMW_XCHG, true, true},
    {TargetOpcode::G_ATOMICRMW_ADD, true, true},
    {TargetOpcode::G_ATOMICRMW_SUB, true, true},
    {TargetOpcode::G_ATOMICRMW_AND, true, true},
    {TargetOpcode::G_ATOMICRMW_OR, true, true},
    {TargetOpcode::G_ATOMICRMW_XOR, true, true},
    {TargetOpcode::G_ATOMICRMW_MIN, true, true},
    {TargetOpcode::G_ATOMICRMW_MAX, true, true},
    {TargetOpcode::G_ATOMICRMW_UMIN, true, true},
    {TargetOpcode::G_ATOMICRMW_UMAX, true, true},
    {TargetOpcode::G_BRJT, true, true},
    {TargetOpcode::G_ROTR, true, true},
    {TargetOpcode::G_ROTL, true, true},
    {TargetOpcode::G_SBFX, true, true},
    {TargetOpcode::G_UBFX, true, true},
    {TargetOpcode::G_SADDSAT, true, true},
    {TargetOpcode::G_SSUBSAT, true, true},
    {TargetOpcode::G_LROUND, false, true},
    {TargetOpcode::G_LLROUND, false, true}};

/// Advanced SIMD intrinsics from IntrinsicsAArch64.td.
/// The list is incomplete. 14. July 2023. CRC and crypto is missing.
static constexpr std::array<unsigned, 500> FloatIntrinsics = {
    Intrinsic::aarch64_neon_faddv,        Intrinsic::aarch64_neon_saddlv,
    Intrinsic::aarch64_neon_uaddlv,       Intrinsic::aarch64_neon_shadd,
    Intrinsic::aarch64_neon_uhadd,        Intrinsic::aarch64_neon_srhadd,
    Intrinsic::aarch64_neon_urhadd,       Intrinsic::aarch64_neon_sqadd,
    Intrinsic::aarch64_neon_suqadd,       Intrinsic::aarch64_neon_usqadd,
    Intrinsic::aarch64_neon_uqadd,        Intrinsic::aarch64_neon_addhn,
    Intrinsic::aarch64_neon_raddhn,       Intrinsic::aarch64_neon_sqdmulh,
    Intrinsic::aarch64_neon_sqdmulh_lane, Intrinsic::aarch64_neon_sqdmulh_laneq,
    Intrinsic::aarch64_neon_sqrdmlah,     Intrinsic::aarch64_neon_sqrdmlsh,
    Intrinsic::aarch64_neon_pmul,         Intrinsic::aarch64_neon_smull,
    Intrinsic::aarch64_neon_umull,        Intrinsic::aarch64_neon_pmull,
    Intrinsic::aarch64_neon_fmulx,        Intrinsic::aarch64_neon_sqdmull,
    Intrinsic::aarch64_neon_shsub,        Intrinsic::aarch64_neon_uhsub,
    Intrinsic::aarch64_neon_sqsub,        Intrinsic::aarch64_neon_uhsub,
    Intrinsic::aarch64_neon_rsubhn,       Intrinsic::aarch64_neon_facge,
    Intrinsic::aarch64_neon_facgt,        Intrinsic::aarch64_neon_sabd,
    Intrinsic::aarch64_neon_uabd,         Intrinsic::aarch64_neon_fabd,
    Intrinsic::aarch64_neon_fabd,         Intrinsic::aarch64_neon_smax,
    Intrinsic::aarch64_neon_fmax,         Intrinsic::aarch64_neon_fmaxnmp,
    Intrinsic::aarch64_neon_fmaxv,        Intrinsic::aarch64_neon_fmaxnmv,
    Intrinsic::aarch64_neon_smin,         Intrinsic::aarch64_neon_umin,
    Intrinsic::aarch64_neon_fmin,         Intrinsic::aarch64_neon_fminnmp,
    Intrinsic::aarch64_neon_fminnm,       Intrinsic::aarch64_neon_fmaxnm,
    Intrinsic::aarch64_neon_sminv,        Intrinsic::aarch64_neon_fmaxv,
    Intrinsic::aarch64_neon_fmaxnmv,      Intrinsic::aarch64_neon_smin,
    Intrinsic::aarch64_neon_umin,         Intrinsic::aarch64_neon_fmin,
    Intrinsic::aarch64_neon_fminnmp,      Intrinsic::aarch64_neon_fminnm,
    Intrinsic::aarch64_neon_fmaxnm,       Intrinsic::aarch64_neon_uminv,
    Intrinsic::aarch64_neon_fminv,        Intrinsic::aarch64_neon_fminnmv,
    Intrinsic::aarch64_neon_addp,         Intrinsic::aarch64_neon_faddp,
    Intrinsic::aarch64_neon_saddlp,       Intrinsic::aarch64_neon_uaddlp,
    Intrinsic::aarch64_neon_smaxp,        Intrinsic::aarch64_neon_umaxp,
    Intrinsic::aarch64_neon_fmaxp,        Intrinsic::aarch64_neon_sminp,
    Intrinsic::aarch64_neon_uminp,        Intrinsic::aarch64_neon_fminp,
    Intrinsic::aarch64_neon_frecps,       Intrinsic::aarch64_neon_frsqrts,
    Intrinsic::aarch64_neon_frecpx,       Intrinsic::aarch64_neon_sqshl,
    Intrinsic::aarch64_neon_uqshl,        Intrinsic::aarch64_neon_srshl,
    Intrinsic::aarch64_neon_urshl,        Intrinsic::aarch64_neon_sqrshl,
    Intrinsic::aarch64_neon_uqrshl,       Intrinsic::aarch64_neon_sqshlu,
    Intrinsic::aarch64_neon_sqshrun,      Intrinsic::aarch64_neon_sqrshrun,
    Intrinsic::aarch64_neon_sqshrn,       Intrinsic::aarch64_neon_uqshrn,
    Intrinsic::aarch64_neon_rshrn,        Intrinsic::aarch64_neon_sqrshrn,
    Intrinsic::aarch64_neon_uqrshrn,      Intrinsic::aarch64_neon_sshl,
    Intrinsic::aarch64_neon_ushl,         Intrinsic::aarch64_neon_shll,
    Intrinsic::aarch64_neon_sshll,        Intrinsic::aarch64_neon_ushll,
    Intrinsic::aarch64_neon_vsri,         Intrinsic::aarch64_neon_vsli,
    Intrinsic::aarch64_neon_sqxtn,        Intrinsic::aarch64_neon_uqxtn,
    Intrinsic::aarch64_neon_sqxtun,       Intrinsic::aarch64_neon_abs,
    Intrinsic::aarch64_neon_sqabs,        Intrinsic::aarch64_neon_sqneg,
    Intrinsic::aarch64_neon_cls,          Intrinsic::aarch64_neon_urecpe,
    Intrinsic::aarch64_neon_frecpe,       Intrinsic::aarch64_neon_ursqrte,
    Intrinsic::aarch64_neon_frsqrte,      Intrinsic::aarch64_neon_fcvtas,
    Intrinsic::aarch64_neon_fcvtau,       Intrinsic::aarch64_neon_fcvtms,
    Intrinsic::aarch64_neon_fcvtmu,       Intrinsic::aarch64_neon_fcvtns,
    Intrinsic::aarch64_neon_fcvtnu,       Intrinsic::aarch64_neon_fcvtps,
    Intrinsic::aarch64_neon_fcvtpu,       Intrinsic::aarch64_neon_fcvtzs,
    Intrinsic::aarch64_neon_fcvtzu,       Intrinsic::aarch64_neon_frint32x,
    Intrinsic::aarch64_neon_frint32z,     Intrinsic::aarch64_neon_frint64x,
    Intrinsic::aarch64_neon_frint64z,     Intrinsic::aarch64_neon_fcvtxn,
    Intrinsic::aarch64_neon_udot,         Intrinsic::aarch64_neon_sdot,
    Intrinsic::aarch64_neon_ummla,        Intrinsic::aarch64_neon_smmla,
    Intrinsic::aarch64_neon_usmmla,       Intrinsic::aarch64_neon_usdot,
    Intrinsic::aarch64_neon_bfmmla,       Intrinsic::aarch64_neon_bfmlalb,
    Intrinsic::aarch64_neon_bfmlalt,      Intrinsic::aarch64_neon_bfcvtn,
    Intrinsic::aarch64_neon_bfcvtn2,      Intrinsic::aarch64_neon_fmlal,
    Intrinsic::aarch64_neon_fmlsl,        Intrinsic::aarch64_neon_fmlal2,
    Intrinsic::aarch64_neon_fmlsl2,       Intrinsic::aarch64_neon_vcadd_rot90,
    Intrinsic::aarch64_neon_vcadd_rot270, Intrinsic::aarch64_neon_vcmla_rot0,
    Intrinsic::aarch64_neon_vcmla_rot90,  Intrinsic::aarch64_neon_vcmla_rot180,
    Intrinsic::aarch64_neon_vcmla_rot270, Intrinsic::aarch64_neon_vcopy_lane,
    Intrinsic::aarch64_neon_ld1x2,        Intrinsic::aarch64_neon_ld1x3,
    Intrinsic::aarch64_neon_ld1x4,        Intrinsic::aarch64_neon_st1x2,
    Intrinsic::aarch64_neon_st1x3,        Intrinsic::aarch64_neon_st1x4,
    Intrinsic::aarch64_neon_ld2,          Intrinsic::aarch64_neon_ld3,
    Intrinsic::aarch64_neon_ld4,          Intrinsic::aarch64_neon_ld2lane,
    Intrinsic::aarch64_neon_ld3lane,      Intrinsic::aarch64_neon_ld4lane,
    Intrinsic::aarch64_neon_ld2r,         Intrinsic::aarch64_neon_ld3r,
    Intrinsic::aarch64_neon_ld4r,         Intrinsic::aarch64_neon_st2,
    Intrinsic::aarch64_neon_st3,          Intrinsic::aarch64_neon_st4,
    Intrinsic::aarch64_neon_st2lane,      Intrinsic::aarch64_neon_st3lane,
    Intrinsic::aarch64_neon_st4lane,      Intrinsic::aarch64_neon_tbl1,
    Intrinsic::aarch64_neon_tbl2,         Intrinsic::aarch64_neon_tbl3,
    Intrinsic::aarch64_neon_tbl4,         Intrinsic::aarch64_neon_tbx1,
    Intrinsic::aarch64_neon_tbx2,         Intrinsic::aarch64_neon_tbx3,
    Intrinsic::aarch64_neon_tbx4,
};

static std::optional<UseDefBehavior> findFloat(unsigned Opcode) {
  // FIXME slow:
  for (auto FloatOp : FloatingPoint)
    if (FloatOp.Opcode == Opcode)
      return FloatOp;
  return std::nullopt;
}

static std::optional<UseDefBehavior> findGPR(unsigned Opcode) {
  // FIXME slow:
  for (auto GPROp : Integer)
    if (GPROp.Opcode == Opcode)
      return GPROp;
  return std::nullopt;
}

AArch64RegBankSelect::AArch64RegBankSelect()
    : RegBankSelect(AArch64RegBankSelect::ID, Mode::Fast) {}

char AArch64RegBankSelect::ID = 0;

/// Returns true when \p MI uses the FPR register bank.
bool AArch64RegBankSelect::usesFPRRegisterBank(
    const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();

  // Check if we already know the register bank.
  auto *RB = MRI.getRegBankOrNull(MI.getOperand(0).getReg());
  if (RB == &AArch64::FPRRegBank)
    return true;
  return false;
}

/// Returns true when \p MI uses the GPRR register bank.
bool AArch64RegBankSelect::usesGPRRegisterBank(
    const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();

  // Check if we already know the register bank.
  auto *RB = MRI.getRegBankOrNull(MI.getOperand(0).getReg());
  if (RB == &AArch64::GPRRegBank)
    return true;
  return false;
}

/// Return whether \p MI uses the GPR register bank
bool AArch64RegBankSelect::usesGPR(const llvm::MachineInstr &MI) const {
  if (auto useDef = findGPR(MI.getOpcode()))
    return useDef->uses;
  return false;
}

/// Return whether \p MI defs the GPR register bank
bool AArch64RegBankSelect::defsGPR(const llvm::MachineInstr &MI) const {
  if (auto useDef = findGPR(MI.getOpcode()))
    return useDef->defs;
  return false;
}

RegisterBankKind
AArch64RegBankSelect::classifyExtract(const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  // For 128 bit sources we have to use FPR unless proven otherwise
  Register Src = MI.getOperand(1).getReg();
  LLT SrcTy = MRI.getType(MI.getOperand(1).getReg());
  if (SrcTy.getSizeInBits() == 128)
    return RegisterBankKind::FPR;
  if (MRI.getRegClassOrNull(Src) == &AArch64::XSeqPAirsClassRegClass)
    return RegisterBankKind::GPR;
  return RegisterBankKind::FPR;
}

RegisterBankKind
AArch64RegBankSelect::classifyBuildVector(const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  Register Src = MI.getOperand(1).getReg();

  if (MRI.getRegClassOrNull(Src) != &AArch64::PMI_FirstGPR)
    return RegisterBankKind::FPR;

  Register VReg = MI.getOperand(1).getReg();
  if (!VReg)
    return RegisterBankKind::GPR; // FIXME

  MachineInstr *DefMI = MRI.getVRegDef(VReg);
  unsigned DefOpc = DefMI->getOpcode();
  LLT SrcTy = MRI.getType(VReg);
  if (all_of(MI.operands(), [&](const MachineOperand &Op) {
        return Op.isDef() || MRI.getVRegDef(Op.getReg())->getOpcode() ==
                                 TargetOpcode::G_CONSTANT;
      })) {
    return RegisterBankKind::GPR;
  }

  if (isFloatingPoint(*DefMI) || SrcTy.getSizeInBits() < 32 ||
      MRI.getRegClassOrNull(VReg) == &AArch64::FPRRegBank) {
    return RegisterBankKind::FPR;
  }

  return RegisterBankKind::FPR; // FIXME
}

// FIXME: refine

/// Classify load or store instruction \p MI.
RegisterBankKind
AArch64RegBankSelect::classifyMemoryDef(const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  LLT Ty = MRI.getType(MI.getOperand(0).getReg());
  unsigned Size = Ty.getSizeInBits();

  if (Size > 64)
    return RegisterBankKind::FPR;

  if (auto *LdSt = dyn_cast<GLoadStore>(&MI)) {
    if (LdSt->isAtomic())
      return RegisterBankKind::GPR;
    LdSt->getMMO().getAlign();
  }

  if (auto *Ld = dyn_cast<GLoad>(&MI)) {
    // Try to guess the type of the Load/Store.
    const auto &MMO = **MI.memoperands_begin();
    const Value *LdVal = MMO.getValue();
    if (LdVal) {
      Type *EltTy = nullptr;
      if (const GlobalValue *GV = dyn_cast<GlobalValue>(LdVal)) {
        EltTy = GV->getValueType();
      } else {
        for (const auto *LdUser : LdVal->users()) {
          if (isa<LoadInst>(LdUser)) {
            EltTy = LdUser->getType();
            break;
          }
          if (isa<StoreInst>(LdUser) && LdUser->getOperand(1) == LdVal) {
            EltTy = LdUser->getType();
            break;
          }
        }
      }
      if (EltTy && EltTy->isFPOrFPVectorTy())
        return RegisterBankKind::FPR;
    }

    // Check if that load feeds fp instructions.
    // In that case, we want the default mapping to be on FPR
    // instead of blind map every scalar to GPR.
    if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
               [&](const MachineInstr &UseMI) {
                 // If we have at least one direct use in a FP instruction,
                 // assume this was a floating point load in the IR. If it was
                 // not, we would have had a bitcast before reaching that
                 // instruction.
                 //
                 // Int->FP conversion operations are also captured in;
                 // defsFPR().
                 return usesFPR(UseMI) || defsFPR(UseMI);
               }))
      return RegisterBankKind::FPR;

    if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
               [&](const MachineInstr &UseMI) {
                 return usesGPR(UseMI) || defsGPR(UseMI);
               }))
      return RegisterBankKind::GPR;
  }

  if (auto *St = dyn_cast<GStore>(&MI)) {
    Register Dst = MI.getOperand(0).getReg();

    if (MRI.getRegClassOrNull(Dst) == &AArch64::PMI_FirstGPR) {
    }
    return RegisterBankKind::FPR;
  }

  // last resort: unknown
  return RegisterBankKind::GPROrFPR;
}

RegisterBankKind
AArch64RegBankSelect::classifyAtomicDef(const llvm::MachineInstr &MI) const {
  // Atomics always use GPR destinations.
  return RegisterBankKind::GPR;
}

RegisterBankKind
AArch64RegBankSelect::classifyCopy(const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  // Check if one of the register is not a generic register.
  if ((DstReg.isPhysical() || !MRI.getType(DstReg).isValid()) ||
      (SrcReg.isPhysical() || !MRI.getType(SrcReg).isValid())) {
    const RegisterBank *DstRB = MRI.getRegBankOrNull(DstReg);
    const RegisterBank *SrcRB = MRI.getRegBankOrNull(SrcReg);
    if (!DstRB)
      DstRB = SrcRB;
    else if (!SrcRB)
      SrcRB = DstRB;
    // If both RB are null that means both registers are generic.
    // We shouldn't be here.
    assert(DstRB && SrcRB && "Both RegBank were nullptr");
    LLT Ty = MRI.getType(DstReg);
    unsigned Size = Ty.getSizeInBits();

    if (DstRB == &AArch64::GPRRegBank)
      return RegisterBankKind::GPR;
    return RegisterBankKind::FPR;
  }

  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
  if (!DstTy.isVector() && DstTy.getSizeInBits() <= 64)
    return RegisterBankKind::GPR;
  return RegisterBankKind::FPR;
}

RegisterBankKind
AArch64RegBankSelect::classifyIntrinsicDef(const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  if (isFloatingPointIntrinsic(MI))
    return RegisterBankKind::FPR;

  // FPR?
  if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
             [&](const MachineInstr &UseMI) {
               return usesFPR(UseMI) || defsFPR(UseMI);
             }))
    return RegisterBankKind::FPR;

  // GPR?
  if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
             [&](const MachineInstr &UseMI) {
               return usesGPR(UseMI) || defsGPR(UseMI);
             }))
    return RegisterBankKind::GPR;

  // FIXME: atomic multi GPR
  LLT DestTy = MRI.getType(MI.getOperand(0).getReg());
  if (DestTy.getScalarSizeInBits() > 64)
    return RegisterBankKind::FPR;

  // last resort
  return RegisterBankKind::GPROrFPR;
}

/// Returns whether instr \p MI is a  floating-point,
/// having only floating-point operands.
bool AArch64RegBankSelect::isFloatingPoint(const llvm::MachineInstr &MI) const {
  if (isFloatingPointIntrinsic(MI))
    return true;
  if (auto FloatOp = findFloat(MI.getOpcode()))
    return FloatOp->uses && FloatOp->defs;
  return false;
}

// FIXME : refine
RegisterBankKind
AArch64RegBankSelect::getDefRegisterBank(const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  LLT Ty = MRI.getType(MI.getOperand(0).getReg());

  bool IsFPR = Ty.isVector() || isFloatingPoint(MI);

  if (IsFPR)
    return RegisterBankKind::FPR;
  return RegisterBankKind::GPR;
}

/// \returns true if a given intrinsic only uses and defines FPRs.
bool AArch64RegBankSelect::isFloatingPointIntrinsic(
    const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  for (auto Intrin : FloatIntrinsics)
    if (Intrin == cast<GIntrinsic>(MI).getIntrinsicID())
      return true;

  switch (cast<GIntrinsic>(MI).getIntrinsicID()) {
  default:
    return false;
  case Intrinsic::aarch64_neon_uaddlv:
  case Intrinsic::aarch64_neon_uaddv:
  case Intrinsic::aarch64_neon_umaxv:
  case Intrinsic::aarch64_neon_uminv:
  case Intrinsic::aarch64_neon_fmaxv:
  case Intrinsic::aarch64_neon_fminv:
  case Intrinsic::aarch64_neon_fmaxnmv:
  case Intrinsic::aarch64_neon_fminnmv:
    return true;
  case Intrinsic::aarch64_neon_saddlv: {
    const LLT SrcTy = MRI.getType(MI.getOperand(2).getReg());
    return SrcTy.getElementType().getSizeInBits() >= 16 &&
           SrcTy.getElementCount().getFixedValue() >= 4;
  }
  case Intrinsic::aarch64_neon_saddv:
  case Intrinsic::aarch64_neon_smaxv:
  case Intrinsic::aarch64_neon_sminv: {
    const LLT SrcTy = MRI.getType(MI.getOperand(2).getReg());
    return SrcTy.getElementType().getSizeInBits() >= 32 &&
           SrcTy.getElementCount().getFixedValue() >= 2;
  }
  }
}

/// Assing \p MI to the FPR register bank
void AArch64RegBankSelect::assignFPR(const llvm::MachineInstr &MI) {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  MRI.setRegBank(MI.getOperand(0).getReg(), AArch64::FPRRegBank);
}

/// Assing \p MI to the GPR register bank
void AArch64RegBankSelect::assignGPR(const llvm::MachineInstr &MI) {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  MRI.setRegBank(MI.getOperand(0).getReg(), AArch64::GPRRegBank);
}

/// Assign unambiguous \p MI to a register bank.
void AArch64RegBankSelect::assignUnambiguousRegisterBank(
    const llvm::MachineInstr &MI) {
  switch (classifyDef(MI)) {
  case RegisterBankKind::GPR:
    assignGPR(MI);
    break;
  case RegisterBankKind::FPR:
    assignFPR(MI);
    break;
  default:
    break;
  }
}

/// Assign ambiguous \p MI to a register bank. It uses heuristics.
void AArch64RegBankSelect::assignAmbiguousRegisterBank(
    const llvm::MachineInstr &MI) {
  const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();

  // uses FPR?
  if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
             [&](MachineInstr &MI) { return usesFPRRegisterBank(MI); })) {
    assignFPR(MI);
    return;
  }

  // uses GPR?
  if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
             [&](MachineInstr &MI) { return usesGPRRegisterBank(MI); })) {
    assignGPR(MI);
    return;
  }

  // FIXME: domain reassignment

  // last resort
  assignGPR(MI);
}

bool AArch64RegBankSelect::isAmbiguous(const llvm::MachineInstr &mi) const {
  return classifyDef(mi) == RegisterBankKind::GPROrFPR;
}

bool AArch64RegBankSelect::isUnambiguous(const llvm::MachineInstr &mi) const {
  return classifyDef(mi) != RegisterBankKind::GPROrFPR;
}

// FIXME: improve

/// Categorizes \p MI in RegisterBankKind. It is the main
/// classifier. The goals are coverage and precision.
RegisterBankKind
AArch64RegBankSelect::classifyDef(const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  LLT Ty = MRI.getType(MI.getOperand(0).getReg());
  unsigned Size = Ty.getSizeInBits();

  bool IsFPR = false;
  if (Ty.isVector() || Size > 64)
    IsFPR = true;

  switch (MI.getOpcode()) {
  case TargetOpcode::G_ANYEXT:
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_SEXT_INREG:
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_TRUNC:
  case TargetOpcode::G_INTTOPTR:
  case TargetOpcode::G_PTRTOINT:
    return getDefRegisterBank(MI);
  case TargetOpcode::G_BITCAST: {
    LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
    if (!DstTy.isVector() && DstTy.getSizeInBits() <= 64)
      return RegisterBankKind::GPR;
    return RegisterBankKind::FPR;
  }
  case TargetOpcode::G_CONSTANT:
    return RegisterBankKind::GPR;
  case TargetOpcode::G_FCONSTANT:
    return RegisterBankKind::FPR;
  case TargetOpcode::G_CTLZ:
  case TargetOpcode::G_CTLZ_ZERO_UNDEF:
  case TargetOpcode::G_CTTZ:
  case TargetOpcode::G_CTTZ_ZERO_UNDEF:
  case TargetOpcode::G_CTPOP: {
    if (Size != 32 || Size != 64)
      return RegisterBankKind::FPR;

    return RegisterBankKind::GPROrFPR;
  }
  case TargetOpcode::G_BSWAP:
  case TargetOpcode::G_BITREVERSE:
  case TargetOpcode::G_ADDRSPACE_CAST:
  case TargetOpcode::G_FREEZE:
    return getDefRegisterBank(MI);
  case TargetOpcode::G_LROUND:
  case TargetOpcode::G_LLROUND:
    return RegisterBankKind::GPR;
  // integer
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_SUB:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_UDIV:
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_UREM:
  case TargetOpcode::G_SDIVREM:
  case TargetOpcode::G_UDIVREM:
    return getDefRegisterBank(MI);
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
    return getDefRegisterBank(MI);
  case TargetOpcode::G_SHL:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_ASHR: {
    LLT ShiftAmtTy = MRI.getType(MI.getOperand(2).getReg());
    LLT SrcTy = MRI.getType(MI.getOperand(1).getReg());
    if (ShiftAmtTy.getSizeInBits() == 64 && SrcTy.getSizeInBits() == 32)
      return RegisterBankKind::GPR;
    return getDefRegisterBank(MI);
  }
  case TargetOpcode::G_FSHL:
  case TargetOpcode::G_FSHR:
    return getDefRegisterBank(MI);
  case TargetOpcode::G_ROTR:
  case TargetOpcode::G_ROTL:
    return RegisterBankKind::GPR;
  case TargetOpcode::G_ICMP:
    return getDefRegisterBank(MI);
  case TargetOpcode::G_FCMP: {
    // vector cmp must be in FPR
    if (Ty.isVector())
      return RegisterBankKind::FPR;
    return RegisterBankKind::GPR;
  }
  case TargetOpcode::G_SELECT: {
    LLT SrcTy = MRI.getType(MI.getOperand(2).getReg());
    if (SrcTy.isVector())
      return RegisterBankKind::FPR;
    if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
               [&](MachineInstr &MI) { return usesFPR(MI); }))
      return RegisterBankKind::FPR;
    // FIXME
    return RegisterBankKind::GPROrFPR;
  }
  case TargetOpcode::G_PTR_ADD:
  case TargetOpcode::G_PTRMASK:
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_SMAX:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_UMAX:
  case TargetOpcode::G_ABS:
  case TargetOpcode::G_UADDO:
  case TargetOpcode::G_UADDE:
  case TargetOpcode::G_SADDO:
  case TargetOpcode::G_SADDE:
  case TargetOpcode::G_USUBO:
  case TargetOpcode::G_USUBE:
  case TargetOpcode::G_SSUBO:
  case TargetOpcode::G_SSUBE:
  case TargetOpcode::G_UMULO:
  case TargetOpcode::G_SMULO:
  case TargetOpcode::G_UMULH:
  case TargetOpcode::G_SMULH:
  case TargetOpcode::G_UADDSAT:
  case TargetOpcode::G_SADDSAT:
  case TargetOpcode::G_USUBSAT:
  case TargetOpcode::G_SSUBSAT:
  case TargetOpcode::G_USHLSAT:
  case TargetOpcode::G_SSHLSAT:
  case TargetOpcode::G_SMULFIX:
  case TargetOpcode::G_UMULFIX:
  case TargetOpcode::G_SMULFIXSAT:
  case TargetOpcode::G_UMULFIXSAT:
  case TargetOpcode::G_SDIVFIX:
  case TargetOpcode::G_UDIVFIX:
  case TargetOpcode::G_SDIVFIXSAT:
  case TargetOpcode::G_UDIVFIXSAT:
  case TargetOpcode::G_FNEG:
    return getDefRegisterBank(MI);
  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI:
    return RegisterBankKind::GPR;
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP:
    return RegisterBankKind::FPR;
  case TargetOpcode::G_FPEXT:
  case TargetOpcode::G_FPTRUNC:
  case TargetOpcode::G_FABS:
  case TargetOpcode::G_FCOPYSIGN:
  case TargetOpcode::G_FCANONICALIZE:
  case TargetOpcode::G_IS_FPCLASS:
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMAXNUM:
  case TargetOpcode::G_FMINNUM_IEEE:
  case TargetOpcode::G_FMAXNUM_IEEE:
  case TargetOpcode::G_FMINIMUM:
  case TargetOpcode::G_FMAXIMUM:
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FMA:
  case TargetOpcode::G_FMAD:
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FREM:
  case TargetOpcode::G_FPOW:
  case TargetOpcode::G_FPOWI:
  case TargetOpcode::G_FEXP:
  case TargetOpcode::G_FEXP2:
  case TargetOpcode::G_FLOG:
  case TargetOpcode::G_FLOG2:
  case TargetOpcode::G_FLOG10:
  case TargetOpcode::G_FLDEXP:
  case TargetOpcode::G_FFREXP:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FSQRT:
  case TargetOpcode::G_FFLOOR:
  case TargetOpcode::G_FRINT:
  case TargetOpcode::G_FNEARBYINT:
  case TargetOpcode::G_VECREDUCE_SEQ_FADD:
  case TargetOpcode::G_VECREDUCE_SEQ_FMUL:
  case TargetOpcode::G_VECREDUCE_FADD:
  case TargetOpcode::G_VECREDUCE_FMUL:
  case TargetOpcode::G_VECREDUCE_FMAX:
  case TargetOpcode::G_VECREDUCE_FMIN:
  case TargetOpcode::G_VECREDUCE_ADD:
  case TargetOpcode::G_VECREDUCE_MUL:
  case TargetOpcode::G_VECREDUCE_AND:
  case TargetOpcode::G_VECREDUCE_OR:
  case TargetOpcode::G_VECREDUCE_XOR:
  case TargetOpcode::G_VECREDUCE_SMAX:
  case TargetOpcode::G_VECREDUCE_SMIN:
  case TargetOpcode::G_VECREDUCE_UMAX:
  case TargetOpcode::G_VECREDUCE_UMIN:
  case TargetOpcode::G_VECREDUCE_FMAXIMUM:
  case TargetOpcode::G_VECREDUCE_FMINIMUM:
    return RegisterBankKind::FPR;
    // intrinsics
  case TargetOpcode::G_INTRINSIC_FPTRUNC_ROUND:
  case TargetOpcode::G_INTRINSIC_TRUNC:
  case TargetOpcode::G_INTRINSIC_LRINT:
  case TargetOpcode::G_INTRINSIC_ROUNDEVEN:
  case TargetOpcode::G_READCYCLECOUNTER:
    return getDefRegisterBank(MI);
    // memory
  case TargetOpcode::G_LOAD:
  case TargetOpcode::G_SEXTLOAD:
  case TargetOpcode::G_ZEXTLOAD:
  case TargetOpcode::G_INDEXED_LOAD:
  case TargetOpcode::G_INDEXED_SEXTLOAD:
  case TargetOpcode::G_INDEXED_ZEXTLOAD:
  case TargetOpcode::G_STORE:
  case TargetOpcode::G_INDEXED_STORE:
    return classifyMemoryDef(MI);
  case TargetOpcode::G_ATOMIC_CMPXCHG_WITH_SUCCESS:
  case TargetOpcode::G_ATOMIC_CMPXCHG:
  case TargetOpcode::G_ATOMICRMW_XCHG:
  case TargetOpcode::G_ATOMICRMW_ADD:
  case TargetOpcode::G_ATOMICRMW_SUB:
  case TargetOpcode::G_ATOMICRMW_AND:
  case TargetOpcode::G_ATOMICRMW_NAND:
  case TargetOpcode::G_ATOMICRMW_OR:
  case TargetOpcode::G_ATOMICRMW_XOR:
  case TargetOpcode::G_ATOMICRMW_MAX:
  case TargetOpcode::G_ATOMICRMW_MIN:
  case TargetOpcode::G_ATOMICRMW_UMAX:
  case TargetOpcode::G_ATOMICRMW_UMIN:
  case TargetOpcode::G_ATOMICRMW_FADD:
  case TargetOpcode::G_ATOMICRMW_FSUB:
  case TargetOpcode::G_ATOMICRMW_FMIN:
  case TargetOpcode::G_ATOMICRMW_UINC_WRAP:
  case TargetOpcode::G_ATOMICRMW_UDEC_WRAP:
    return classifyAtomicDef(MI);
  case TargetOpcode::G_FENCE:
    return getDefRegisterBank(MI);
  case TargetOpcode::G_EXTRACT:
    return classifyExtract(MI);
  case TargetOpcode::G_UNMERGE_VALUES: {
    // FIXME: improve
    LLT SrcTy = MRI.getType(MI.getOperand(MI.getNumOperands() - 1).getReg());
    if (SrcTy.isVector() || SrcTy == LLT::scalar(128) ||
        any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
               [&](MachineInstr &MI) { return usesFPR(MI); }))
      return RegisterBankKind::FPR;
    return RegisterBankKind::GPROrFPR;
  }
  case TargetOpcode::G_INSERT:
    return RegisterBankKind::GPR;
  case TargetOpcode::G_MERGE_VALUES: {
    // FIXME: improve
    LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
    if (DstTy.isVector() || DstTy == LLT::scalar(128) ||
        any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
               [&](MachineInstr &MI) { return usesFPR(MI); }))
      return RegisterBankKind::FPR;
    return RegisterBankKind::GPROrFPR;
  }
  case TargetOpcode::G_BUILD_VECTOR:
  case TargetOpcode::G_BUILD_VECTOR_TRUNC:
    return classifyBuildVector(MI);
  case TargetOpcode::G_INTRINSIC:
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS:
  case TargetOpcode::G_INTRINSIC_CONVERGENT:
  case TargetOpcode::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS:
    return classifyIntrinsicDef(MI);
  case TargetOpcode::G_BR:
  case TargetOpcode::G_BRCOND:
  case TargetOpcode::G_BRINDIRECT:
  case TargetOpcode::G_BRJT:
    return RegisterBankKind::GPR;
  case TargetOpcode::G_READ_REGISTER:
  case TargetOpcode::G_WRITE_REGISTER:
    return RegisterBankKind::GPR;
  // vector ops
  case TargetOpcode::G_INSERT_VECTOR_ELT:
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
  case TargetOpcode::G_SHUFFLE_VECTOR:
    return RegisterBankKind::FPR;
  // constrained floating point
  case TargetOpcode::G_STRICT_FADD:
  case TargetOpcode::G_STRICT_FSUB:
  case TargetOpcode::G_STRICT_FMUL:
  case TargetOpcode::G_STRICT_FDIV:
  case TargetOpcode::G_STRICT_FREM:
  case TargetOpcode::G_STRICT_FMA:
  case TargetOpcode::G_STRICT_FSQRT:
  case TargetOpcode::G_STRICT_FLDEXP:
    return RegisterBankKind::FPR;
  // memory
  case TargetOpcode::G_MEMCPY:
  case TargetOpcode::G_MEMCPY_INLINE:
  case TargetOpcode::G_MEMMOVE:
  case TargetOpcode::G_MEMSET:
  case TargetOpcode::G_BZERO:
    return RegisterBankKind::GPR;
    // bit field extraction
  case TargetOpcode::G_SBFX:
  case TargetOpcode::G_UBFX: {
    if (Ty.isVector())
      return RegisterBankKind::FPR;
    return RegisterBankKind::GPR;
  }
  case TargetOpcode::COPY:
    return classifyCopy(MI);
  case AArch64::G_DUP:
    return RegisterBankKind::FPR;
  default: {
    reportGISelFailure(const_cast<MachineFunction &>(MF), *TPC, *MORE,
                       "gisel-aarch64-regbankselect", "failed to classifyDef",
                       MI);
    return RegisterBankKind::GPR;
  }
  }
}

bool AArch64RegBankSelect::isDomainReassignable(
    const llvm::MachineInstr &mi) const {
  switch (mi.getOpcode()) {
  case TargetOpcode::G_OR:
  case TargetOpcode::G_FNEG:
  case TargetOpcode::G_BITCAST:
  case TargetOpcode::G_STORE:
  case TargetOpcode::G_LOAD:
  case TargetOpcode::G_SELECT:
    return true;
  default:
    return false;
  }
}

/// Returns whether \p MI defs FPR bank
bool AArch64RegBankSelect::defsFPR(const llvm::MachineInstr &MI) const {
  if (auto FloatOp = findFloat(MI.getOpcode()))
    return FloatOp->defs;
  return false;
}

/// Returns whether \p MI uses FPR bank
bool AArch64RegBankSelect::usesFPR(const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  if (MI.isPHI())
    return any_of(MI.explicit_uses(), [&](const MachineOperand &Op) {
      return Op.isReg() && defsFPR(*MRI.getVRegDef(Op.getReg()));
    });

  if (auto FloatOp = findFloat(MI.getOpcode()))
    return FloatOp->uses;

  if (MI.getOpcode() == TargetOpcode::G_INTRINSIC) {
    return isFloatingPointIntrinsic(MI);
  } else {
    return false;
  }
}

bool AArch64RegBankSelect::isUnassignable(const MachineInstr &MI) const {
  return (isTargetSpecificOpcode(MI.getOpcode()) && !MI.isPreISelOpcode()) ||
         MI.isInlineAsm() || MI.isDebugInstr();
}

bool AArch64RegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  // If the ISel pipeline failed, do not bother running that pass.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  MORE = std::make_unique<MachineOptimizationRemarkEmitter>(MF, nullptr);

  init(MF);

  // assign unambiguous in any order
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &mi : MBB.instrs()) {
      if (isUnassignable(mi))
        continue;
      if (isUnambiguous(mi))
        assignUnambiguousRegisterBank(mi);
    }

  // uses before defs
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT)
    for (MachineInstr &mi : MBB->instrs()) {
      if (isUnassignable(mi))
        continue;
      if (isAmbiguous(mi))
        assignAmbiguousRegisterBank(mi);
    }

  return true;
}

// https://reviews.llvm.org/D89415

INITIALIZE_PASS_BEGIN(AArch64RegBankSelect, DEBUG_TYPE, "AArch64 RegBankSelect",
                      false, false)
INITIALIZE_PASS_END(AArch64RegBankSelect, DEBUG_TYPE, "AArch64 RegBankSelect",
                    false, false)

namespace llvm {

FunctionPass *createAArch64RegBankSelect() {
  return new AArch64RegBankSelect();
}

} // namespace llvm
