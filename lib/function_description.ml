open Ctypes
module Types = Types_generated

module Functions (F : FOREIGN) = struct
  open F

  let run_main =
    foreign "ireeCompilerRunMain" (int @-> ptr (ptr char) @-> returning int)

  let global_initialize =
    foreign "ireeCompilerGlobalInitialize" (void @-> returning void)

  let session_create =
    foreign "ireeCompilerSessionCreate"
      (void @-> returning (ptr_opt Types.session))

  let session_set_flags =
    foreign "ireeCompilerSessionSetFlags"
      (ptr Types.session @-> int @-> ptr string @-> returning void)

  let source_wrap_buffer =
    foreign "ireeCompilerSourceWrapBuffer"
      ( ptr Types.session @-> string @-> string @-> size_t @-> bool
      @-> ptr (ptr Types.source)
      @-> returning (ptr_opt Types.error) )

  let error_get_message =
    foreign "ireeCompilerErrorGetMessage" (ptr Types.error @-> returning string)

  let error_destroy =
    foreign "ireeCompilerErrorDestroy" (ptr Types.error @-> returning void)

  let session_destroy =
    foreign "ireeCompilerSessionDestroy" (ptr Types.session @-> returning void)

  let output_open_file =
    foreign "ireeCompilerOutputOpenFile"
      (string @-> ptr (ptr Types.output) @-> returning (ptr_opt Types.error))

  let invocation_create =
    foreign "ireeCompilerInvocationCreate"
      (ptr Types.session @-> returning (ptr_opt Types.invocation))

  let invocation_parse_source =
    foreign "ireeCompilerInvocationParseSource"
      (ptr Types.invocation @-> ptr Types.source @-> returning bool)

  let invocation_destroy =
    foreign "ireeCompilerInvocationDestroy"
      (ptr Types.invocation @-> returning void)

  let invocation_pipeline =
    foreign "ireeCompilerInvocationPipeline"
      (ptr Types.invocation @-> Types.pipeline @-> returning bool)

  let invocation_output_vm_bytecode =
    foreign "ireeCompilerInvocationOutputVMBytecode"
      ( ptr Types.invocation @-> ptr Types.output
      @-> returning (ptr_opt Types.error) )

  let instance_options_initialize =
    foreign "iree_runtime_instance_options_initialize"
      (ptr Types.instance_options @-> returning void)

  let instance_options_use_all_available_drivers =
    foreign "iree_runtime_instance_options_use_all_available_drivers"
      (ptr Types.instance_options @-> returning void)

  let allocator_system =
    foreign "iree_allocator_system" (void @-> returning Types.allocator)

  let instance_create =
    foreign "iree_runtime_instance_create"
      ( ptr Types.instance_options @-> Types.allocator
      @-> ptr (ptr Types.instance)
      @-> returning Types.status )

  let instance_release =
    foreign "iree_runtime_instance_release"
      (ptr Types.instance @-> returning void)

  let status_ok = foreign "iree_status_is_ok" (Types.status @-> returning bool)

  let status_to_string =
    foreign "iree_status_to_string"
      ( Types.status @-> ptr Types.allocator
      @-> ptr (ptr char)
      @-> ptr size_t @-> returning void )

  let make_string_view =
    foreign "iree_make_cstring_view" (string @-> returning Types.string_view)

  let instance_try_create_default_device =
    foreign "iree_runtime_instance_try_create_default_device"
      ( ptr Types.instance @-> Types.string_view
      @-> ptr (ptr Types.device)
      @-> returning Types.status )

  let device_release =
    foreign "iree_hal_device_release" (ptr Types.device @-> returning void)

  let session_options_initialize =
    foreign "iree_runtime_session_options_initialize"
      (ptr Types.session_options @-> returning void)

  let instance_host_allocator =
    foreign "iree_runtime_instance_host_allocator"
      (ptr Types.instance @-> returning Types.allocator)

  let session_create_with_device =
    foreign "iree_runtime_session_create_with_device"
      ( ptr Types.instance @-> ptr Types.session_options @-> ptr Types.device
      @-> Types.allocator
      @-> ptr (ptr Types.runtime_session)
      @-> returning Types.status )

  let session_release =
    foreign "iree_runtime_session_release"
      (ptr Types.runtime_session @-> returning void)

  let session_append_bytecode_module_from_file =
    foreign "iree_runtime_session_append_bytecode_module_from_file"
      (ptr Types.runtime_session @-> string @-> returning Types.status)

  let call_initialize_by_name =
    foreign "iree_runtime_call_initialize_by_name"
      ( ptr Types.runtime_session @-> Types.string_view @-> ptr Types.call
      @-> returning Types.status )

  let session_device_allocator =
    foreign "iree_runtime_session_device_allocator"
      (ptr Types.runtime_session @-> returning (ptr Types.device_allocator))

  let session_host_allocator =
    foreign "iree_runtime_session_host_allocator"
      (ptr Types.runtime_session @-> returning Types.allocator)

  let make_const_byte_span =
    foreign "iree_make_const_byte_span"
      (ptr void @-> size_t @-> returning Types.const_byte_span)

  let buffer_view_allocate_buffer_copy =
    foreign "iree_hal_buffer_view_allocate_buffer_copy"
      ( ptr Types.device @-> ptr Types.device_allocator @-> size_t
      @-> ptr size_t @-> Types.element_type @-> Types.encoding_type
      @-> Types.buffer_params @-> Types.const_byte_span
      @-> ptr (ptr Types.buffer_view)
      @-> returning Types.status )

  let buffer_view_release =
    foreign "iree_hal_buffer_view_release"
      (ptr Types.buffer_view @-> returning void)

  let call_inputs_push_back_buffer_view =
    foreign "iree_runtime_call_inputs_push_back_buffer_view"
      (ptr Types.call @-> ptr Types.buffer_view @-> returning Types.status)

  let call_invoke =
    foreign "iree_runtime_call_invoke"
      (ptr Types.call @-> uint32_t @-> returning Types.status)

  let call_reset =
    foreign "iree_runtime_call_reset" (ptr Types.call @-> returning void)

  let call_outputs_pop_front_buffer_view =
    foreign "iree_runtime_call_outputs_pop_front_buffer_view"
      (ptr Types.call @-> ptr (ptr Types.buffer_view) @-> returning Types.status)

  let device_transfer_d2h =
    foreign "iree_hal_device_transfer_d2h"
      ( ptr Types.device @-> ptr Types.buffer @-> size_t @-> ptr void @-> size_t
      @-> uint32_t @-> Types.timeout @-> returning Types.status )

  let buffer_view_buffer =
    foreign "iree_hal_buffer_view_buffer"
      (ptr Types.buffer_view @-> returning (ptr Types.buffer))

  let infinite_timeout =
    foreign "iree_infinite_timeout" (void @-> returning Types.timeout)
end
