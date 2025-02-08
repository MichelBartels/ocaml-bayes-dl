open Dsl

type t =
  | Normal of (Ir.Tensor.f32, float) Ir.Var.u * (Ir.Tensor.f32, float) Ir.Var.u
  | Uniform of (Ir.Tensor.f32, float) Ir.Var.u * (Ir.Tensor.f32, float) Ir.Var.u

let sample = function
  | Normal (mean, std) ->
      let shape = Ir.shape_of_var mean in
      let sample = norm ~.0. ~.1. shape in
      (sample *@ std) +@ mean
  | Uniform (low, high) ->
      let shape = Ir.shape_of_var low in
      let sample = uniform ~.0. ~.1. shape in
      (sample *@ (high -@ low)) +@ low

let log_prob dist x =
  match dist with
  | Normal (mean, std) ->
      let shape = Ir.shape_of_var mean in
      let axes = List.mapi (fun i _ -> i) shape in
      let squared_error = (((x -@ mean) /@ std) **.> 2.) /.> 2. in
      let regulariser = Float.log (2. *. Float.pi /. 2.) +.< ln std in
      Dsl.sum axes @@ ~-@(squared_error +@ regulariser)
  | Uniform (low, high) ->
      let shape = Ir.shape_of_var low in
      let axes = List.mapi (fun i _ -> i) shape in
      Dsl.sum axes @@ ln (low -@ high)
