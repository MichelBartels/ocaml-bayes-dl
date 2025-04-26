open Dsl

type t =
  | Normal of (Ir.Tensor.f32, float) Ir.Var.u * (Ir.Tensor.f32, float) Ir.Var.u
  | Uniform of (Ir.Tensor.f32, float) Ir.Var.u * (Ir.Tensor.f32, float) Ir.Var.u

type _ Effect.t +=
  | Sample : t * t * int option -> (Ir.Tensor.f32, float) Ir.Var.u Effect.t

let sample dist batch_size =
  match dist with
  | Normal (mean, std) ->
      let mean, std =
        match batch_size with
        | Some batch_size ->
            let mean = Ir.Var.BroadcastInDim (mean, [batch_size]) in
            let std = Ir.Var.BroadcastInDim (std, [batch_size]) in
            (mean, std)
        | None ->
            (mean, std)
      in
      let shape = Ir.shape_of_var mean in
      let sample = Random.normal_f32 shape in
      (sample *@ std) +@ mean
  | Uniform (low, high) ->
      let low, high =
        match batch_size with
        | Some batch_size ->
            let low = Ir.Var.BroadcastInDim (low, [batch_size]) in
            let high = Ir.Var.BroadcastInDim (high, [batch_size]) in
            (low, high)
        | None ->
            (low, high)
      in
      let shape = Ir.shape_of_var low in
      let sample = Random.uniform_f32 shape in
      (sample *@ (high -@ low)) +@ low

let log_prob ?batch_size dist x =
  match dist with
  | Normal (mean, stddev) ->
      let mean, stddev =
        match batch_size with
        | Some batch_size ->
            let mean = Ir.Var.BroadcastInDim (mean, [batch_size]) in
            let stddev = Ir.Var.BroadcastInDim (stddev, [batch_size]) in
            (mean, stddev)
        | None ->
            (mean, stddev)
      in
      let shape = Ir.shape_of_var mean in
      let axes = List.mapi (fun i _ -> i) shape in
      let scaled_diff = (x -@ mean) /@ stddev in
      let squared_error = scaled_diff *@ scaled_diff /.> 2. in
      let regulariser = (Float.log (2. *. Float.pi) /. 2.) +.< ln stddev in
      Dsl.sum axes @@ ~-@(squared_error +@ regulariser)
  | Uniform (low, high) ->
      let low, high =
        match batch_size with
        | Some batch_size ->
            let low = Ir.Var.BroadcastInDim (low, [batch_size]) in
            let high = Ir.Var.BroadcastInDim (high, [batch_size]) in
            (low, high)
        | None ->
            (low, high)
      in
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
      let var_ratio = std_p /@ std_q in
      let scaled_diff = (mean_p -@ mean_q) /@ std_q in
      let squared_diff = scaled_diff *@ scaled_diff in
      let axes = List.mapi (fun i _ -> i) @@ Ir.shape_of_var mean_p in
      Dsl.sum axes
        ((var_ratio +@ squared_diff -.> 1. +@ ln std_q -@ ln std_p) /.> 2.)
  | _ ->
      failwith "ELBO not implemented for this distribution"
