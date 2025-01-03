open Ctypes

module Types (F : TYPE) = struct
  open F

  type session

  type error

  type source

  type output

  type invocation

  type pipeline = Hal_executable | Precompile | Std

  type instance_options

  type hal_driver_registry

  type instance

  type status_handle

  type allocator

  type allocator_command = Malloc | Calloc | Realloc | Free

  type string_view

  type device

  type session_options

  type runtime_session

  type vm_module

  type vm_function

  type vm_list

  type call

  type device_allocator

  type buffer_view

  type buffer_params

  type const_byte_span

  type buffer

  type timeout_type = Absolute | Relative

  type timeout

  let session : session structure typ = structure "iree_compiler_session_t"

  let error : error structure typ = structure "iree_compiler_error_t"

  let source : source structure typ = structure "iree_compiler_source_t"

  let output : output structure typ = structure "iree_compiler_output_t"

  let invocation : invocation structure typ =
    structure "iree_compiler_invocation_t"

  let compiler_pipeline_std = constant "IREE_COMPILER_PIPELINE_STD" int64_t

  let compiler_pipeline_hal_executable =
    constant "IREE_COMPILER_PIPELINE_HAL_EXECUTABLE" int64_t

  let compiler_pipeline_precompile =
    constant "IREE_COMPILER_PIPELINE_PRECOMPILE" int64_t

  let pipeline =
    enum "iree_compiler_pipeline_t"
      [ (Std, compiler_pipeline_std)
      ; (Hal_executable, compiler_pipeline_hal_executable)
      ; (Precompile, compiler_pipeline_precompile) ]

  let instance_options : instance_options structure typ =
    structure "iree_runtime_instance_options_t"

  let hal_driver_registry : hal_driver_registry structure typ =
    structure "iree_hal_driver_registry_t"

  let _ = field instance_options "driver_registry" (ptr hal_driver_registry)

  let () = seal instance_options

  let instance : instance structure typ = structure "iree_runtime_instance_t"

  let status_handle : status_handle structure typ =
    structure "iree_status_handle_t"

  let status = ptr status_handle

  let allocator : allocator structure typ = structure "iree_allocator_t"

  let _ = field allocator "self" (ptr void)

  let allocator_command_malloc =
    constant "IREE_ALLOCATOR_COMMAND_MALLOC" int64_t

  let allocator_command_calloc =
    constant "IREE_ALLOCATOR_COMMAND_CALLOC" int64_t

  let allocator_command_realloc =
    constant "IREE_ALLOCATOR_COMMAND_REALLOC" int64_t

  let allocator_command_free = constant "IREE_ALLOCATOR_COMMAND_FREE" int64_t

  let allocator_command =
    enum "iree_allocator_command_e"
      [ (Malloc, allocator_command_malloc)
      ; (Calloc, allocator_command_calloc)
      ; (Realloc, allocator_command_realloc)
      ; (Free, allocator_command_free) ]

  let allocator_ctl_fn =
    static_funptr
      ( ptr void @-> allocator_command @-> ptr void
      @-> ptr (ptr void)
      @-> returning void )

  let _ = field allocator "ctl" allocator_ctl_fn

  let () = seal allocator

  let string_view : string_view structure typ = structure "iree_string_view_t"

  let _ = field string_view "data" string

  let _ = field string_view "size" size_t

  let () = seal string_view

  let device : device structure typ = structure "iree_hal_device_t"

  let session_options : session_options structure typ =
    structure "iree_runtime_session_options_t"

  let _ = field session_options "context_flags" uint32_t

  let _ = field session_options "builtin_modules" uint64_t

  let () = seal session_options

  let runtime_session : runtime_session structure typ =
    structure "iree_runtime_session_t"

  let vm_module : vm_module structure typ = structure "iree_vm_module_t"

  let vm_function : vm_function structure typ = structure "iree_vm_function_t"

  let _ = field vm_function "module" (ptr vm_module)

  let _ = field vm_function "linkage" uint16_t

  let _ = field vm_function "ordinal" uint16_t

  let () = seal vm_function

  let vm_list : vm_list structure typ = structure "iree_vm_list_t"

  let call : call structure typ = structure "iree_runtime_call_t"

  let _ = field call "session" (ptr runtime_session)

  let _ = field call "function" vm_function

  let _ = field call "inputs" (ptr vm_list)

  let _ = field call "outputs" (ptr vm_list)

  let () = seal call

  let device_allocator : device_allocator structure typ =
    structure "iree_hal_allocator_t"

  let buffer_view : buffer_view structure typ =
    structure "iree_hal_buffer_view_t"

  let element_type_int_32 = constant "IREE_HAL_ELEMENT_TYPE_INT_32" uint32_t

  let element_type_sint_32 = constant "IREE_HAL_ELEMENT_TYPE_SINT_32" uint32_t

  let element_type_uint_32 = constant "IREE_HAL_ELEMENT_TYPE_UINT_32" uint32_t

  let element_type_int_64 = constant "IREE_HAL_ELEMENT_TYPE_INT_64" uint32_t

  let element_type_sint_64 = constant "IREE_HAL_ELEMENT_TYPE_SINT_64" uint32_t

  let element_type_uint_64 = constant "IREE_HAL_ELEMENT_TYPE_UINT_64" uint32_t

  let element_type_float_16 = constant "IREE_HAL_ELEMENT_TYPE_FLOAT_16" uint32_t

  let element_type_float_32 = constant "IREE_HAL_ELEMENT_TYPE_FLOAT_32" uint32_t

  let element_type_float_64 = constant "IREE_HAL_ELEMENT_TYPE_FLOAT_64" uint32_t

  let element_type_bfloat_16 =
    constant "IREE_HAL_ELEMENT_TYPE_BFLOAT_16" uint32_t

  let element_type_bool_8 = constant "IREE_HAL_ELEMENT_TYPE_BOOL_8" uint32_t

  let element_type = uint32_t

  let encoding_type_dense_row_major =
    constant "IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR" uint32_t

  let encoding_type = uint32_t

  let buffer_params : buffer_params structure typ =
    structure "iree_hal_buffer_params_t"

  let buffer_usage_default = constant "IREE_HAL_BUFFER_USAGE_DEFAULT" uint32_t

  let buffer_usage = field buffer_params "usage" uint32_t

  let memory_access_all = constant "IREE_HAL_MEMORY_ACCESS_ALL" uint16_t

  let buffer_access = field buffer_params "access" uint16_t

  let memory_type_device_local =
    constant "IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL" uint32_t

  let buffer_memory_type = field buffer_params "type" uint32_t

  let _ = field buffer_params "queue_affinity" uint64_t

  let _ = field buffer_params "min_alignment" size_t

  let () = seal buffer_params

  let const_byte_span : const_byte_span structure typ =
    structure "iree_const_byte_span_t"

  let _ = field const_byte_span "data" (ptr uint8_t)

  let _ = field const_byte_span "data_length" size_t

  let () = seal const_byte_span

  let buffer : buffer structure typ = structure "iree_hal_buffer_t"

  let timeout_type =
    enum "iree_timeout_type_e"
      [ (Absolute, constant "IREE_TIMEOUT_ABSOLUTE" int64_t)
      ; (Relative, constant "IREE_TIMEOUT_RELATIVE" int64_t) ]

  let timeout : timeout structure typ = structure "iree_timeout_t"

  let _ = field timeout "type" timeout_type

  let _ = field timeout "nanos" int64_t

  let () = seal timeout

  let transfer_buffer_flag_default =
    constant "IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT" uint32_t
end
