add_executable(regbank
           main.cpp
           )

target_include_directories(regbank PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(regbank
        PRIVATE
        regbankselect
#       ${llvm_libs}
)


