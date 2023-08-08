#include "AArch64RegisterBankInfoV2.h"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/CodeGen/GlobalISel/GenericMachineInstrs.h>
#include <llvm/CodeGen/GlobalISel/Utils.h>
#include <llvm/CodeGen/MachineBlockFrequencyInfo.h>
#include <llvm/CodeGen/MachineBranchProbabilityInfo.h>
#include <llvm/CodeGen/MachineOptimizationRemarkEmitter.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/RegisterBankInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/IntrinsicsAArch64.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;
using OperandsMapper = llvm::RegisterBankInfo::OperandsMapper;

const unsigned DefaultMappingID = UINT_MAX;

#define DEBUG_TYPE "AArch64RegBankSelectV2"

// clang-format off
// clang-format on

bool AArch64RegisterBankInfoV2::usesFRPRegisterBank(
    const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();

  // Check if we already know the register bank.
  auto *RB = getRegBank(MI.getOperand(0).getReg(), MRI, TRI);
  if (RB == &AArch64::FPRRegBank)
    return true;
  // if (RB == &AArch64::GPRRegBank)
  return false;
}

RegisterBankKind AArch64RegisterBankInfoV2::classifyMemoryDef(
    const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  LLT Ty = MRI.getType(MI.getOperand(0).getReg());
  unsigned Size = Ty.getSizeInBits();

  if (Size > 64)
    return RegisterBankKind::FPR;

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
               // Int->FP conversion operations are also captured in
               // onlyDefinesFP().
               return usesFPR(UseMI) || onlyDefinesFP(UseMI, MRI, TRI);
             }))
    return RegisterBankKind::FPR;

  // last resort: unknown
  return RegisterBankKind::GPROrFPR;
}

RegisterBankKind AArch64RegisterBankInfoV2::classifyAtomicDef(
    const llvm::MachineInstr &MI) const {
  // Atomics always use GPR destinations.
  return RegisterBankKind::GPR;
}

RegisterBankKind AArch64RegisterBankInfoV2::classifyIntrinsicDef(
    const llvm::MachineInstr &MI) const {}

/// Returns whether instr \p MI is a  floating-point,
/// having only floating-point operands.
bool AArch64RegisterBankInfoV2::isFloatingPoint(
    const llvm::MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FMA:
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FCONSTANT:
  case TargetOpcode::G_FPEXT:
  case TargetOpcode::G_FPTRUNC:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FFLOOR:
  case TargetOpcode::G_FNEARBYINT:
  case TargetOpcode::G_FNEG:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FLOG10:
  case TargetOpcode::G_FLOG:
  case TargetOpcode::G_FLOG2:
  case TargetOpcode::G_FSQRT:
  case TargetOpcode::G_FABS:
  case TargetOpcode::G_FEXP:
  case TargetOpcode::G_FRINT:
  case TargetOpcode::G_INTRINSIC_TRUNC:
  case TargetOpcode::G_INTRINSIC_ROUND:
  case TargetOpcode::G_FMAXNUM:
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMAXIMUM:
  case TargetOpcode::G_FMINIMUM:
    return true;
  }
  return false;
}

RegisterBankKind AArch64RegisterBankInfoV2::getDefRegisterBank(
    const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  // unsigned NumOperands = MI.getNumOperands();

  LLT Ty = MRI.getType(MI.getOperand(0).getReg());
  // unsigned Size = Ty.getSizeInBits();

  //  if (Size > 64)
  //    return RegisterBankKind::FPR;

  bool IsFPR = Ty.isVector() || isFloatingPoint(MI);

  if (IsFPR)
    return RegisterBankKind::FPR;
  return RegisterBankKind::GPR;
}

unsigned AArch64RegisterBankInfoV2::copyCost(const llvm::RegisterBank &A,
                                             const llvm::RegisterBank &B,
                                             unsigned Size) const {
  // What do we do with different size?
  // copy are same size.
  // Will introduce other hooks for different size:
  // * extract cost.
  // * build_sequence cost.

  // Copy from (resp. to) GPR to (resp. from) FPR involves FMOV.
  // FIXME: This should be deduced from the scheduling model.
  if (&A == &AArch64::GPRRegBank && &B == &AArch64::FPRRegBank)
    // FMOVXDr or FMOVWSr.
    return 5;
  if (&A == &AArch64::FPRRegBank && &B == &AArch64::GPRRegBank)
    // FMOVDXr or FMOVSWr.
    return 4;

  return RegisterBankInfo::copyCost(A, B, Size);
}

void AArch64RegisterBankInfoV2::assignFPR(const llvm::MachineInstr &MI) {
  const unsigned Opc = MI.getOpcode();
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
  LLT SrcTy = MRI.getType(MI.getOperand(1).getReg());
  unsigned Size = DstTy.getSizeInBits();
  bool DstIsGPR = !DstTy.isVector() && DstTy.getSizeInBits() <= 64;
  bool SrcIsGPR = !SrcTy.isVector() && SrcTy.getSizeInBits() <= 64;

  const RegisterBank &DstRB =
      DstIsGPR ? AArch64::GPRRegBank : AArch64::FPRRegBank;
  const RegisterBank &SrcRB =
      SrcIsGPR ? AArch64::GPRRegBank : AArch64::FPRRegBank;

  const RegisterBankInfo::InstructionMapping &InstrMapping =
      getInstructionMapping(
          DefaultMappingID, copyCost(DstRB, *SrcRB, Size),
          getCopyMapping(DstRB.getID(), SrcRB.getID(), Size),
          // We only care about the mapping of the destination.
          /*NumOperands*/ MI.getNumOperands());

  applyMapping(MI, InstrMapping, {});
}

void AArch64RegisterBankInfoV2::assignGPR(const llvm::MachineInstr &MI) {
  applyMapping(MI, InstrMapping, {});
}

void AArch64RegisterBankInfoV2::assignUnambiguousRegisterBank(
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

void AArch64RegisterBankInfoV2::assignAmbiguousRegisterBank(
    const llvm::MachineInstr &MI) {
  const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();

  //  float blockFreq = MBFI->getBlockFreqRelativeToEntryBlock(
  //      &MI.getParent()->getParent()->front());

  // uses FPR?
  if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
             [&](MachineInstr &MI) { return usesFRPRegisterBank(MI); })) {
    assignFPR(MI);
    return;
  }

  // domain reassignment

  // last resort
  assignGPR(MI);
}

bool AArch64RegisterBankInfoV2::isAmbiguous(
    const llvm::MachineInstr &mi) const {
  return classifyDef(mi) == RegisterBankKind::GPROrFPR;
}

bool AArch64RegisterBankInfoV2::isUnambiguous(
    const llvm::MachineInstr &mi) const {
  return classifyDef(mi) != RegisterBankKind::GPROrFPR;
}

RegisterBankKind
AArch64RegisterBankInfoV2::classifyDef(const llvm::MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  // const TargetRegisterInfo &TRI = *STI.getRegisterInfo();
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
    xxx if (Size != 32 && Size != 64) return RegisterBankKind::FPR;

    return RegisterBankKind::GPROrFPR;
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
    return RegisterBankKind::GPR;
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
    return getDefRegisterBank(MI);
  case TargetOpcode::G_SHL:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_ASHR: {
    LLT ShiftAmtTy = MRI.getType(MI.getOperand(2).getReg());
    LLT SrcTy = MRI.getType(MI.getOperand(1).getReg());
  }

  case TargetOpcode::G_FSHL:
  case TargetOpcode::G_FSHR:
  case TargetOpcode::G_ROTR:
  case TargetOpcode::G_ROTL:
  case TargetOpcode::G_ICMP:

  case TargetOpcode::G_FCMP: {
    // vector cmp must be in FPR
    if (Ty.isVector())
      return RegisterBankKind::FPR;
    return RegisterBankKind::GPR;
  }
  case TargetOpcode::G_SELECT:
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

  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI:

  // float
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP:
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
    return RegisterBankKind::FPR;

    // intrinsics
  case TargetOpcode::G_INTRINSIC_FPTRUNC_ROUND:
  case TargetOpcode::G_INTRINSIC_TRUNC:
  case TargetOpcode::G_INTRINSIC_LRINT:
  case TargetOpcode::G_INTRINSIC_ROUNDEVEN:
  case TargetOpcode::G_READCYCLECOUNTER:

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
  case TargetOpcode::G_EXTRACT:
    return RegisterBankKind::FPR;
  case TargetOpcode::G_UNMERGE_VALUES: {
    xxxx LLT SrcTy =
        MRI.getType(MI.getOperand(MI.getNumOperands() - 1).getReg());
    if (SrcTy.isVector() || SrcTy == LLT::scalar(128) ||
            any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg())),
        [&](MachineInstr &MI) { return usesFPR(MI); })
      return RegisterBankKind::FPR;
    return RegisterBankKind::GPROrFPR;
  }
  case TargetOpcode::G_INSERT:
  case TargetOpcode::G_MERGE_VALUES:
  case TargetOpcode::G_BUILD_VECTOR:

  case TargetOpcode::G_BUILD_VECTOR_TRUNC:

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
  case TargetOpcode::G_INVOKE_REGION_START:
  case TargetOpcode::G_READ_REGISTER:
  case TargetOpcode::G_WRITE_REGISTER:

  // vector ops
  case TargetOpcode::G_INSERT_VECTOR_ELT:
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
    return REgisterBankKind::FPR;
  case TargetOpcode::G_SHUFFLE_VECTOR:

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
  case TargetOpcode::COPY: {
    xxx
  }
  default: {
    MORE->emit([&] {
      MachineOptimizationRemarkMissed R(DEBUG_TYPE, "gisel-reg-bank-select2",
                                        MI.getDebugLoc(), /*MMB*/ nullptr);

      std::string MIAsString;
      raw_string_ostream Stream = raw_string_ostream(MIAsString);
      MI.print(Stream);
      R << "failed to classifyDef: " << MIAsString << ".";
    });
    return RegisterBankKind::GPR;
  }
  }
}

bool AArch64RegisterBankInfoV2::isDomainReassignable(
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

bool RegBankSelect2::defsFPR(const llvm::MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case AArch64::G_DUP:
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP:
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
  case TargetOpcode::G_INSERT_VECTOR_ELT:
  case TargetOpcode::G_BUILD_VECTOR:
  case TargetOpcode::G_BUILD_VECTOR_TRUNC:
    return true;
  default:
    break;
  }
  xxx
}

bool AArch64RegisterBankInfoV2::usesFPR(const llvm::MachineInstr &MI) const {
  if (M.isPHI())
    return any_of(MI.explicit_uses(), [&](const MachineOperand &Op) {
      return Op.Reg() && defsFPR(*MRI.getVRegDef(Op.getReg()));
    });
  switch (MI.getOpcode()) {
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FMA:
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FCONSTANT:
  case TargetOpcode::G_FPEXT:
  case TargetOpcode::G_FPTRUNC:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FFLOOR:
  case TargetOpcode::G_FNEARBYINT:
  case TargetOpcode::G_FNEG:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FLOG10:
  case TargetOpcode::G_FLOG:
  case TargetOpcode::G_FLOG2:
  case TargetOpcode::G_FSQRT:
  case TargetOpcode::G_FABS:
  case TargetOpcode::G_FEXP:
  case TargetOpcode::G_FRINT:
  case TargetOpcode::G_INTRINSIC_TRUNC:
  case TargetOpcode::G_INTRINSIC_ROUND:
  case TargetOpcode::G_FMAXNUM:
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMAXIMUM:
  case TargetOpcode::G_FMINIMUM:
    return true;
  case TargetOpcode::G_INTRINSIC: {
    // TODO: Add more intrinsics.
    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
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
  default:
    return false;
  }
}

void AArch64RegisterBankInfoV2::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

const RegisterBank &
AArch64RegisterBankInfoV2::getRegBankFromRegClass(const TargetRegisterClass &RC,
                                                  LLT) const {
  switch (RC.getID()) {
  case AArch64::FPR8RegClassID:
  case AArch64::FPR16RegClassID:
  case AArch64::FPR16_loRegClassID:
  case AArch64::FPR32_with_hsub_in_FPR16_loRegClassID:
  case AArch64::FPR32RegClassID:
  case AArch64::FPR64RegClassID:
  case AArch64::FPR64_loRegClassID:
  case AArch64::FPR128RegClassID:
  case AArch64::FPR128_loRegClassID:
  case AArch64::DDRegClassID:
  case AArch64::DDDRegClassID:
  case AArch64::DDDDRegClassID:
  case AArch64::QQRegClassID:
  case AArch64::QQQRegClassID:
  case AArch64::QQQQRegClassID:
    return getRegBank(AArch64::FPRRegBankID);
  case AArch64::GPR32commonRegClassID:
  case AArch64::GPR32RegClassID:
  case AArch64::GPR32spRegClassID:
  case AArch64::GPR32sponlyRegClassID:
  case AArch64::GPR32argRegClassID:
  case AArch64::GPR32allRegClassID:
  case AArch64::GPR64commonRegClassID:
  case AArch64::GPR64RegClassID:
  case AArch64::GPR64spRegClassID:
  case AArch64::GPR64sponlyRegClassID:
  case AArch64::GPR64argRegClassID:
  case AArch64::GPR64allRegClassID:
  case AArch64::GPR64noipRegClassID:
  case AArch64::GPR64common_and_GPR64noipRegClassID:
  case AArch64::GPR64noip_and_tcGPR64RegClassID:
  case AArch64::tcGPR64RegClassID:
  case AArch64::rtcGPR64RegClassID:
  case AArch64::WSeqPairsClassRegClassID:
  case AArch64::XSeqPairsClassRegClassID:
  case AArch64::MatrixIndexGPR32_8_11RegClassID:
  case AArch64::MatrixIndexGPR32_12_15RegClassID:
  case AArch64::GPR64_with_sub_32_in_MatrixIndexGPR32_8_11RegClassID:
  case AArch64::GPR64_with_sub_32_in_MatrixIndexGPR32_12_15RegClassID:
    return getRegBank(AArch64::GPRRegBankID);
  case AArch64::CCRRegClassID:
    return getRegBank(AArch64::CCRegBankID);
  default:
    llvm_unreachable("Register class not supported");
  }
}

bool AArch64RegisterBankInfoV2::isUnassignable(const MachineInstr &MI) const {
  return (isTargetSpecificOpcode(MI.getOpcode()) && !MI.isPreISelOpcode()) ||
         MI.isInlineAsm() || MI.isDebugInstr();
}

bool AArch64RegisterBankInfoV2::runOnMachineFunction(MachineFunction &MF) {
  MORE = std::make_unique<MachineOptimizationRemarkEmitter>(MF, MBFI);

  init(MF);

  // assign unambiguous any order
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &mi : reverse(MBB.instrs())) {
      if (isUnassignable(mi)
          continue;
      if (isUnambiguous(mi))
        assignUnambiguousRegisterBank(mi);
    }

  // uses before defs?
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT)
    for (MachineInstr &mi : reverse(MBB->instrs())) { // FIXME ???
      if (isUnassignable(mi)
          continue;
      if (isAmbiguous(mi))
        assignAmbiguousRegisterBank(mi);
    }

  return true;
}