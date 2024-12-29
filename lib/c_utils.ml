open Ctypes

let create_out_param err_fn c_type f =
  let out_ptr = allocate_n c_type ~count:1 in
  let err = f out_ptr in
  err_fn err ; !@out_ptr

let protect f x = Gc.finalise f x ; x
