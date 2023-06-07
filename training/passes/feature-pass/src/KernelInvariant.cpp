#include <llvm/ADT/Optional.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include "AnalysisUtils.hpp"
#include "KernelInvariant.hpp"

using namespace llvm;
using namespace celerity;

KernelInvariant::KernelInvariant(llvm::Function& fun) : function(&fun)
{
  // 1. check the arguments
  for (unsigned i = 0; i < fun.arg_size(); i++) {
    llvm::Argument* arg = fun.getArg(i);
    // if we have a pointer, it cannot be used for loop bound analysis, thus we skip it
    if (arg->getType()->isPointerTy())
      continue;
    InvariantType inv = InvariantType(i);
    invariants[inv] = arg;
  }

  // 2. check all call invocatoin to get_global_size, get_global_local_size, ... and subgroup related
  for (BasicBlock& bb : fun.getBasicBlockList()) {
    for (Instruction& inst : bb) {
      Instruction* inst_ptr = &inst;
      if (auto* ci = llvm::dyn_cast<llvm::CallInst>(&inst)) { // we assume direct call here
        // handling a function call
        std::string fnd = get_demangled_name(*ci->getCalledFunction());
        // outs() << " operands num "<< ci->getNumOperands() << "\n";
        if (ci->getNumOperands() == 2) { // ret + first in op
          Value* operand = ci->getOperand(0);
          if (fnd.rfind("get_global_size", 0) == 0) { // we expect something like "get_global_size(0)
            if (llvm::ConstantInt* int_op = dyn_cast<llvm::ConstantInt>(operand)) {
              if (int_op->getBitWidth() <= 32) {
                int dim = int_op->getSExtValue();
                switch (dim) {
                case 0:
                  invariants[InvariantType::gs0] = inst_ptr;
                  break;
                case 1:
                  invariants[InvariantType::gs1] = inst_ptr;
                  break;
                case 2:
                  invariants[InvariantType::gs2] = inst_ptr;
                  break;
                }
              }
            } else
              errs() << "  warning: operand for get_global_size not recognized\n";
          }                                          // get_global_size
          if (fnd.rfind("get_local_size", 0) == 0) { // we expect something like "get_local_size(0)
            if (llvm::ConstantInt* int_op = dyn_cast<llvm::ConstantInt>(operand)) {
              if (int_op->getBitWidth() <= 32) {
                int dim = int_op->getSExtValue();
                switch (dim) {
                case 0:
                  invariants[InvariantType::ls0] = inst_ptr;
                  break;
                case 1:
                  invariants[InvariantType::ls1] = inst_ptr;
                  break;
                case 2:
                  invariants[InvariantType::ls2] = inst_ptr;
                  break;
                }
              }
            } else
              errs() << "  warning: operand for get_local_size not recognized\n";
          }                                          // get_local_size
          if (fnd.rfind("get_num_groups", 0) == 0) { // we expect something like "get_global_size(0)
            if (llvm::ConstantInt* int_op = dyn_cast<llvm::ConstantInt>(operand)) {
              if (int_op->getBitWidth() <= 32) {
                int dim = int_op->getSExtValue();
                switch (dim) {
                case 0:
                  invariants[InvariantType::ng0] = inst_ptr;
                  break;
                case 1:
                  invariants[InvariantType::ng1] = inst_ptr;
                  break;
                case 2:
                  invariants[InvariantType::ng2] = inst_ptr;
                  break;
                }
              }
            } else
              errs() << "  warning: operand for get_num_groups not recognized\n";
          } // get_num_groups
        }

        if (ci->getNumOperands() == 1) {
          if (fnd.rfind("get_num_sub_groups", 0) == 0)
            invariants[InvariantType::nsg] = inst_ptr;
          if (fnd.rfind("get_sub_group_size", 0) == 0)
            invariants[InvariantType::sgs] = inst_ptr;
          if (fnd.rfind("get_max_sub_group_size", 0) == 0)
            invariants[InvariantType::msgs] = inst_ptr;
        }
      }
    }
  }
} // end ctor

std::map<InvariantType, llvm::Value*> KernelInvariant::getInvariants() { return invariants; }

InvariantType KernelInvariant::isInvariant(Value* value)
{
  for (std::map<InvariantType, llvm::Value*>::iterator it = invariants.begin(); it != invariants.end(); ++it) {
    Value* invariant_value = it->second;
    if (invariant_value == value) { // if is the same variable
      return it->first;
    }
    for (auto use = invariant_value->use_begin(); use != invariant_value->use_end(); use++) { // check all uses of the variable
      if (*use == value) {
        return it->first;
      }
    }
  }
  return InvariantType::none;
}

void KernelInvariant::print(llvm::raw_ostream& out_stream)
{
  out_stream.changeColor(llvm::raw_null_ostream::Colors::GREEN, true);
  out_stream << "kernel invariants: ";
  out_stream.changeColor(llvm::raw_null_ostream::Colors::WHITE, false);
  for (std::map<InvariantType, llvm::Value*>::iterator it = invariants.begin(); it != invariants.end(); ++it) {
    out_stream << "";
    out_stream << InvariantTypeName[it->first];
    out_stream << " ";
  }
  out_stream << "\n";
}
