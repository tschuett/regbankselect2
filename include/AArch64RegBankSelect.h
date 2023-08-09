//===- AArch64RegBankSelect --------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the RegBankSelect class for AArch64.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64REGBANKSELECT_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64REGBANKSELECT_H

#include <llvm/CodeGen/GlobalISel/RegBankSelect.h>

namespace llvm {

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

/*
 * Algorithm
 * 1. Categorize MachineInstr in RegisterBankKind as complete and precise as
 * possible. Unknowns are modeled with Or.
 * 2. Assign register banks for unambiguous MachineInstr.
 * 3. Use assigned register banks to assign ambiguous MachineInstr to register
 * banks.
 *
 * Claim: In the third step, there are only a few MIs and a lot of assigned
 * register banks.
 */

class AArch64RegBankSelect : public llvm::RegBankSelect {
  // classifiers
  RegisterBankKind classifyDef(const llvm::MachineInstr &) const;
  RegisterBankKind classifyMemoryDef(const llvm::MachineInstr &MI) const;
  RegisterBankKind classifyAtomicDef(const llvm::MachineInstr &MI) const;
  RegisterBankKind classifyIntrinsicDef(const llvm::MachineInstr &MI) const;
  RegisterBankKind getDefRegisterBank(const llvm::MachineInstr &MI) const;

  // predicates
  bool isDomainReassignable(const llvm::MachineInstr &) const;
  bool usesFPR(const llvm::MachineInstr &) const;
  bool defsFPR(const llvm::MachineInstr &) const;
  bool isUnambiguous(const llvm::MachineInstr &) const;
  bool isAmbiguous(const llvm::MachineInstr &) const;
  bool isFloatingPoint(const llvm::MachineInstr &) const;
  bool usesFRPRegisterBank(const llvm::MachineInstr &MI) const;
  bool isUnassignable(const MachineInstr &MI) const;

  /// assignment
  void assignAmbiguousRegisterBank(const llvm::MachineInstr &);
  void assignUnambiguousRegisterBank(const llvm::MachineInstr &);
  void assignFPR(const llvm::MachineInstr &MI);
  void assignGPR(const llvm::MachineInstr &MI);


  unsigned copyCost(const llvm::RegisterBank &A, const llvm::RegisterBank &B,
                    unsigned Size) const;

public:
  static char ID;

  AArch64RegBankSelect();

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  llvm::StringRef getPassName() const override {
    return "AArch64RegBankSelectV2";
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  llvm::MachineFunctionProperties getRequiredProperties() const override {
    return llvm::MachineFunctionProperties()
        .set(llvm::MachineFunctionProperties::Property::IsSSA)
        .set(llvm::MachineFunctionProperties::Property::Legalized);
  }

private:
  /// Current optimization remark emitter. Used to report failures.
  std::unique_ptr<llvm::MachineOptimizationRemarkEmitter> MORE;
};

llvm::MachineFunctionPass *createAArch64RegBankSelect() {
  return new AArch64RegBankSelect();
}

} // namespace llvm

#endif