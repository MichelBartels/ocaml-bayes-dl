open Dsl

let rotate x n =
  x >>.> Unsigned.UInt64.of_int n |@ (x <<.> Unsigned.UInt64.of_int (64 - n))

let squares32 ctr key =
  let x = ctr *@ key in
  let y = x in
  let z = y +@ key in
  let x = (x *@ x) +@ y in
  let x = rotate x 32 in
  let x = (x *@ x) +@ z in
  let x = rotate x 32 in
  let x = (x *@ x) +@ y in
  (x *@ x) +@ z >>.> Unsigned.UInt64.of_int 32

let random_u64_to_f32 x =
  let x = x >>.> Unsigned.UInt64.of_int 9 in
  let x = x |.> Unsigned.UInt64.of_string "0x3f800000" in
  let f = x |> convert U32 |> bitcast F32 in
  f -.> 1.0

let key = scalar_u64 "0xc8e4fd154ce32f6d"

let uniform_f32 ?(key = key) shape =
  let total_size = List.fold_left ( * ) 1 shape in
  let ctr = Effect.perform (Effects.Counter total_size) in
  let flat_shape = [total_size] in
  let ctrs = iota 0 flat_shape in
  let ctr = broadcast_scalar ctr flat_shape in
  let ctrs = ctrs +@ ctr in
  let key = broadcast_scalar key flat_shape in
  let x = squares32 ctrs key |> random_u64_to_f32 in
  let x = reshape shape x in
  no_grad x

let erfinv x =
  let sign = select (x <.> 0.0) (full_like F32 (-1.0) x) (full_like F32 1. x) in
  let x = (1.0 -.< x) *@ (1.0 +.< x) in
  let x = ln x in
  let tt1 = (2.0 /. (Float.pi *. 0.147)) +.< (0.5 *.< x) in
  let tt2 = 1.0 /. 0.147 *.< x in
  (sign *@ tt1 *@ tt1) -@ tt2

(* let normal_f32 ?(key = key) shape = *)
(*   let size = List.fold_left ( * ) 1 shape in *)
(*   let half_size = size / 2 in *)
(*   assert (size mod 2 = 0) ; *)
(*   let flat_shape = [half_size] in *)
(*   let u0 = uniform_f32 ~key flat_shape in *)
(*   let u1 = uniform_f32 ~key flat_shape in *)
(*   let factor = sqrt (-2.0 *.< ln (1. -.< u0)) in *)
(*   let inner = 2.0 *. Float.pi *.< u1 in *)
(*   let z_0 = factor *@ sin inner in *)
(*   let z_1 = factor *@ cos inner in *)
(*   concat 0 [z_0; z_1] |> reshape shape |> no_grad *)

let normal_f32 ?(key = key) shape =
  Float.sqrt 2. *.< erfinv ((uniform_f32 ~key shape *.> 2.0) -.> 1.0)

let current_seed () = Effect.perform (Effects.Counter 0)

let handler f (ctr : (Ir.Tensor.u64, Unsigned.uint64) Ir.Var.u) =
  let open Effect.Deep in
  let ctr_ref = ref ctr in
  try_with f ()
    { effc=
        (fun (type a) (eff : a Effect.t) ->
          match eff with
          | Effects.Counter incr ->
              Some
                (fun (k : (a, _) continuation) ->
                  let ctr = !ctr_ref in
                  ctr_ref := ctr +@ scalar_u64 (string_of_int incr) ;
                  continue k ctr )
          | _ ->
              None ) }

let seed_type = ([], Ir.Tensor.U64)

let initial_seed = scalar_u64 "0"
