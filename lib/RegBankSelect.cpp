#include "RegBankSelect.h"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/CodeGen/GlobalISel/GenericMachineInstrs.h>
#include <llvm/CodeGen/GlobalISel/Utils.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/RegisterBankInfo.h>
#include <llvm/IR/IntrinsicsAArch64.h>

using namespace llvm;
using OperandsMapper = llvm::RegisterBankInfo::OperandsMapper;

const unsigned DefaultMappingID = UINT_MAX;

void RegBankSelect2::applyFPR(const llvm::MachineInstr &MI) {
  const unsigned Opc = MI.getOpcode();
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  unsigned NumOperands = MI.getNumOperands();

  LLT Ty = MRI.getType(MI.getOperand(0).getReg());
  unsigned Size = Ty.getSizeInBits();

  const OperandsMapper &OpdMapper = getInstructionMapping(
      DefaultMappingID, 1, getValueMapping(PMI_FirstFPR, Size), NumOperands);

  for (unsigned OpIdx = 0,
                EndIdx = OpdMapper.getInstrMapping().getNumOperands();
       OpIdx != EndIdx; ++OpIdx) {
    const MachineOperand &MO = MI.getOperand(OpIdx);
    if (!MO.isReg()) {
      continue;
    }
    if (!MO.getReg()) {
      continue;
    }
    LLT Ty = MRI.getType(MO.getReg());
    if (!Ty.isValid())
      continue;
  }
}

void RegBankSelect2::assignUnambiguousRegisterBank(const llvm::MachineInstr &) {
}

void RegBankSelect2::assignAmbiguousRegisterBank(const llvm::MachineInstr &MI) {
  const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();

  // uses FPR?
  if (any_of(MRI.use_nodbg_instructions(MI.getOperand(0).getReg()),
             [&](MachineInstr &MI) { return usesFPR(MI); })) {
    // assign FP
    return;
  }

  // last resort
  // assign GPR
}

bool RegBankSelect2::isAmbiguous(const llvm::MachineInstr &mi) {
  return classifyDef(mi) == RegisterBankKind::GPROrFPR;
}

bool RegBankSelect2::isUnambiguous(const llvm::MachineInstr &mi) {
  return classifyDef(mi) != RegisterBankKind::GPROrFPR;
}

RegisterBankKind RegBankSelect2::classifyDef(const llvm::MachineInstr &mi) {
  switch (mi.getOpcode()) {
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
  default:
    return RegisterBankKind::GPR;
  }
}

bool RegBankSelect2::isDomainReassignable(const llvm::MachineInstr &mi) {
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

bool RegBankSelect2::usesFPR(const llvm::MachineInstr &MI) {
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

bool RegBankSelect2::runOnMachineFunction(MachineFunction &MF) {

  // const TargetSubtargetInfo &STI = MF.getSubtarget();

  // assign unambiguous any order
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &mi : reverse(MBB.instrs())) {
      if (isUnambiguous(mi))
        assignUnambiguousRegisterBank(mi);
    }
  }

  // uses before defs?
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT) {
    for (MachineInstr &mi : reverse(MBB->instrs())) {
      if (isAmbiguous(mi))
        assignAmbiguousRegisterBank(mi);
    }
  }

  return true;
}
