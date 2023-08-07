#pragma once

#include <llvm/CodeGen/MachineFunctionPass.h>

#include <llvm/CodeGen/RegisterBankInfo.h>

enum RegisterBankKind {
  // R0 - R15
  GPR,
  // D0 - D31
  FPR,
  // R or D registers
  GPROrFPR,
  // Z0 - Z31
  SVEData,
  // P0 - P15
  SVEPredicate
};

class RegBankSelect2 : public llvm::MachineFunctionPass {
  RegisterBankKind classifyDef(const llvm::MachineInstr &);
  bool isDomainReassignable(const llvm::MachineInstr &);
  bool usesFPR(const llvm::MachineInstr &);
  bool isUnambiguous(const llvm::MachineInstr &);
  bool isAmbiguous(const llvm::MachineInstr &);

  void assignAmbiguousRegisterBank(const llvm::MachineInstr &);
  void assignUnambiguousRegisterBank(const llvm::MachineInstr &);

  void applyFPR(const llvm::MachineInstr &MI);
public:
  RegBankSelect2(char &ID) : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;
};
