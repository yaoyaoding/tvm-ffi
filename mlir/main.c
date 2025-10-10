struct TVMFFIAny {
    int32_t type_index;
    int32_t padding;
    union {
        int32_t v_int;
        float v_float;
        void* v_ptr;
    } v;
};
extern void TVMFFIErrorSetRaisedFromCStr(const char* kind, const char* message);

int __tvm_ffi_add_one_c(
    void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result
  ) {
    if(num_args != 2) {
        TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor input");
    }
    if(args[0].type_index != 1) {
        return -1
    }
}
