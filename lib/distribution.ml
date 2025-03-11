open Dsl

type t =
  | Normal of (Ir.Tensor.f32, float) Ir.Var.u * (Ir.Tensor.f32, float) Ir.Var.u
  | Uniform of (Ir.Tensor.f32, float) Ir.Var.u * (Ir.Tensor.f32, float) Ir.Var.u

type _ Effect.t += Sample : t * t -> (Ir.Tensor.f32, float) Ir.Var.u Effect.t

let sample = function
  | Normal (mean, std) ->
      let shape = Ir.shape_of_var mean in
      let sample = Random.normal_f32 shape in
      (sample *@ std) +@ mean
  | Uniform (low, high) ->
      let shape = Ir.shape_of_var low in
      let sample = Random.uniform_f32 shape in
      (sample *@ (high -@ low)) +@ low

let log_prob dist x =
  match dist with
  | Normal (mean, var) ->
      let shape = Ir.shape_of_var mean in
      let axes = List.mapi (fun i _ -> i) shape in
      let squared_error = (x -@ mean) /@ var *@ ((x -@ mean) /@ var) /.> 2. in
      let regulariser = (Float.log (2. *. Float.pi) /. 2.) +.< ln var in
      Dsl.sum axes @@ ~-@(squared_error +@ regulariser)
  | Uniform (low, high) ->
      let shape = Ir.shape_of_var low in
      let axes = List.mapi (fun i _ -> i) shape in
      Dsl.sum axes @@ ~-@(ln (high -@ low))

let expectation dist =
  match dist with
  | Normal (mean, _) ->
      mean
  | Uniform (low, high) ->
      (low +@ high) /.> 2.

let kl p q =
  (* p = guide q = prior *)
  match (p, q) with
  | Normal (mean_p, std_p), Normal (mean_q, std_q) ->
      let axes = List.mapi (fun i _ -> i) @@ Ir.shape_of_var mean_p in
      Dsl.sum axes
      @@ ln (std_q /@ std_p)
         +@ ((std_p *@ std_p) +@ ((mean_p -@ mean_q) *@ (mean_p -@ mean_q)))
            /@ (2. *.< (std_q *@ std_q))
         -.> 0.5
  | _ ->
      failwith "ELBO not implemented for this distribution"
