open QCheck2
open Gen
open Iree_bindings
open Dsl

let () = Printexc.record_backtrace true

let dim = 4

let num_dims_gen = int_range 1 4

let tensor_gen num_dims =
  let size = int_of_float @@ (float_of_int dim ** float_of_int num_dims) in
  let shape = List.init num_dims (Fun.const dim) in
  let data = List.init size (Fun.const @@ float_range ~-.1. 1.) in
  let* data = flatten_l data in
  return @@ tensor_to_ir @@ Ir.Tensor.of_list F32 shape data

let rec differentiable_function_gen n num_dims =
  let unary_function_gen () =
    let* f, num_dims = differentiable_function_gen (n - 1) num_dims in
    let compose g =
      return
        ( (fun x ->
            let y = f x in
            g y )
        , num_dims )
    in
    frequency
      [ (1, compose ( ~-$ ))
      ; (1, compose ln)
      ; (1, compose exp)
      ; (1, compose ln1p)
      ; (1, compose sqrt)
      ; (1, compose sin)
      ; (1, compose cos)
      ; (1, compose tanh)
      ; (1, compose abs) ]
  in
  let binary_function_gen () =
    let* f1, num_dims1 = differentiable_function_gen (n - 1) num_dims in
    let* f2, num_dims2 = differentiable_function_gen (n - 1) num_dims in
    let min_dims = Int.min num_dims1 num_dims2 in
    let max_dims = Int.max num_dims1 num_dims2 in
    let f1_same_dim, f2_same_dim =
      if num_dims1 = num_dims2 then (f1, f2)
      else
        let broadcast f x =
          let y = f x in
          Ir.Var.BroadcastInDim
            (y, List.init (max_dims - min_dims) (Fun.const dim))
        in
        if num_dims1 > num_dims2 then (f1, broadcast f2) else (broadcast f1, f2)
    in
    let compose g =
      return
        ( (fun x ->
            let y1 = f1_same_dim x in
            let y2 = f2_same_dim x in
            g y1 y2 )
        , max_dims )
    in
    let options = List.map (fun f -> (1, compose f)) [( +$ ); ( *$ ); ( /$ ); ( **$ )] in
    let options =
      if num_dims1 > 1 && num_dims2 > 1 then (1, compose ( @$ )) :: options
      else options
    in
    frequency options
  in
  let reduction_function_gen () =
    let* f, num_dims = differentiable_function_gen (n - 1) num_dims in
    let compose g =
      return
        ( (fun x ->
            let y = f x in
            g y )
        , num_dims )
    in
    frequency
      [ (1, compose (fun x -> sum [0] x))
      ; (1, compose (fun x -> mean [0] x)) ]
  in
  match n with
  | 0 ->
      frequency
        [ (1, return (Fun.id, num_dims))
        ; ( 1
          , let* num_dims = num_dims_gen in
            let* tensor = tensor_gen num_dims in
            return (Fun.const tensor, num_dims) ) ]
  | _ ->
      frequency [(2, unary_function_gen ()); (1, binary_function_gen ()); (1, reduction_function_gen ())]

let differentiable_function_gen =
  (* let* n = int_range 0 1 in *)
  sized_size (int_range 0 2)
  @@ fun n ->
  let* num_dims = int_range 1 4 in
  let* f, num_output_dims = differentiable_function_gen n num_dims in
  return
    ( (fun x ->
        let y = f x in
        let dims = List.init num_dims Fun.id in
        let x_sum = sum dims x in
        let dims = List.init num_output_dims Fun.id in
        sum dims y +$ x_sum )
    , num_dims )

let random_mask num_dims =
  let size = int_of_float @@ (float_of_int dim ** float_of_int num_dims) in
  let shape = List.init num_dims (Fun.const dim) in
  let indices = iota 0 [size] in
  let* index = int_range 0 (size - 1) in
  let index = Unsigned.UInt64.of_int index in
  let index = full U64 index [size] in
  let mask = indices =$ index in
  return @@ reshape shape mask

let finite_difference ?(delta = 1e-4) f mask x =
  let x' = select mask (x +$. delta) x in
  let x'' = select mask (x -$. delta) x in
  let y' = f x' in
  let y'' = f x'' in
  let diff = y' -$ y'' in
  diff /$. (2. *. delta)

let actual_gradient f mask x =
  let [E grad; _] =
    Backpropagate.diff Var (fun (Ir.Var.List.E x) -> Ir.Var.List.E (f x)) (E x)
  in
  let shape = Ir.shape_of_var grad in
  let dims = List.mapi (fun i _ -> i) shape in
  let grad = select mask grad (zeros_like grad) in
  sum dims grad

module Device =
  ( val Pjrt_bindings.make ~caching:false
          "/Users/michelbartels/Downloads/pjrt/jax_plugins/metal_plugin/pjrt_plugin_metal_14.dylib"
    )

let () = Dsl.metal_hack := true

module Runtime = Runtime.Make (Device)

type grad_test =
  { f: (Ir.Tensor.f32, float) Ir.Var.u -> (Ir.Tensor.f32, float) Ir.Var.u
  ; mask: (Ir.Tensor.i1, bool) Ir.Var.u
  ; x: (Ir.Tensor.f32, float) Ir.Var.u }

let grad_test_gen =
  let* f, num_dims = differentiable_function_gen in
  let* mask = random_mask num_dims in
  let* x = tensor_gen num_dims in
  return {f; mask; x}

let grad_test_fn {f; mask; x} Ir.Var.List.[] =
  let grad_actual = actual_gradient f mask x in
  let grad_approx = finite_difference f mask x in
  Ir.Var.List.[E grad_actual; E grad_approx]

let grad_actual_global = ref 0.

let grad_approx_global = ref 0.

let correct_gradient test_state =
  let result = Runtime.compile [] (grad_test_fn test_state) [] in
  let [E grad_actual; E grad_approx] =
    Runtime.DeviceValue.to_host_value result
  in
  let grad_actual = Ir.Tensor.to_list grad_actual in
  let grad_actual = List.hd grad_actual in
  let grad_approx = Ir.Tensor.to_list grad_approx in
  let grad_approx = List.hd grad_approx in
  grad_actual_global := grad_actual ;
  grad_approx_global := grad_approx ;
  (not @@ Float.is_finite grad_approx)
  ||
  let diff = Float.abs (grad_actual -. grad_approx) in
  diff <= (Float.abs grad_actual +. Float.abs grad_approx) *. 1e-1

let print_fn test_state =
  let input_type = Ir.ValueType.of_var test_state.x in
  let func = Ir.create_func (E input_type) (fun (E x) -> E (test_state.f x)) in
  Printf.printf "grad_actual: %f\ngrad_approx: %f\n" !grad_actual_global
    !grad_approx_global ;
  Ir.compile func

let test_grad =
  QCheck2.Test.make ~name:"grad_test" ~count:1 ~print:print_fn grad_test_gen
    correct_gradient

let _ = QCheck_base_runner.run_tests ~verbose:false [test_grad]
