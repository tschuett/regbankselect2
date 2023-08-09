add_library(regbankselect
           AArch64RegBankSelect.cpp
           )


target_include_directories(regbankselect PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs CodeGen Support)


target_link_libraries(
        regbankselect
        PRIVATE
        ${llvm_libs}
        )
