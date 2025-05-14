open Dsl

let gamma x =
  (* Simple approximation of gamma function using Stirling's approximation *)
  let x_minus_half = x -$. 0.5 in
  let sqrt_two_pi = Float.sqrt (2. *. Float.pi) in
  let x_pow = pow x x_minus_half in
  let exp_neg_x = exp ~-$x in
  sqrt_two_pi *.$ x_pow *$ exp_neg_x

let ln_beta alpha beta =
  let gamma_sum = gamma (alpha +$ beta) in
  let gamma_alpha = gamma alpha in
  let gamma_beta = gamma beta in
  let prod = gamma_alpha *$ gamma_beta in
  ln (prod /$ gamma_sum)

let digamma x =
  (* Simple approximation of digamma function *)
  let h = 1e-4 in
  let x_plus_h = x +$. h in
  let x_minus_h = x -$. h in
  let diff = ln (gamma x_plus_h) -$ ln (gamma x_minus_h) in
  diff /$. (2. *. h)

type t =
  | Normal of (Tensor.f32, float) Var.u * (Tensor.f32, float) Var.u
  | Uniform of (Tensor.f32, float) Var.u * (Tensor.f32, float) Var.u
  | Beta of (Tensor.f32, float) Var.u * (Tensor.f32, float) Var.u
  | Exponential of (Tensor.f32, float) Var.u

type _ Effect.t +=
  | Sample : t * t * int option -> (Tensor.f32, float) Var.u Effect.t

let sample dist batch_size =
  match dist with
  | Normal (mean, std) ->
      let mean, std =
        match batch_size with
        | Some batch_size ->
            let mean = Var.BroadcastInDim (mean, [batch_size]) in
            let std = Var.BroadcastInDim (std, [batch_size]) in
            (mean, std)
        | None ->
            (mean, std)
      in
      let shape = Var.shape mean in
      let sample = Random.normal_f32 shape in
      (sample *$ std) +$ mean
  | Uniform (low, high) ->
      let low, high =
        match batch_size with
        | Some batch_size ->
            let low = Var.BroadcastInDim (low, [batch_size]) in
            let high = Var.BroadcastInDim (high, [batch_size]) in
            (low, high)
        | None ->
            (low, high)
      in
      let shape = Var.shape low in
      let sample = Random.uniform_f32 shape in
      (sample *$ (high -$ low)) +$ low
  | Beta (alpha, beta) ->
      let alpha, beta =
        match batch_size with
        | Some batch_size ->
            let alpha = Var.BroadcastInDim (alpha, [batch_size]) in
            let beta = Var.BroadcastInDim (beta, [batch_size]) in
            (alpha, beta)
        | None ->
            (alpha, beta)
      in
      let shape = Var.shape alpha in
      let u1 = Random.uniform_f32 shape in
      let u2 = Random.uniform_f32 shape in
      let x = ~-$(ln u1) /$ alpha in
      let y = ~-$(ln u2) /$ beta in
      x /$ (x +$ y)
  | Exponential rate ->
      let rate =
        match batch_size with
        | Some batch_size ->
            Var.BroadcastInDim (rate, [batch_size])
        | None ->
            rate
      in
      let shape = Var.shape rate in
      let u = Random.uniform_f32 shape in
      ~-$(ln u) /$ rate

let log_prob ?batch_size ?(reduction=(fun x -> sum x)) dist x =
  match dist with
  | Normal (mean, stddev) ->
      let mean, stddev =
        match batch_size with
        | Some batch_size ->
            let mean = Var.BroadcastInDim (mean, [batch_size]) in
            let stddev = Var.BroadcastInDim (stddev, [batch_size]) in
            (mean, stddev)
        | None ->
            (mean, stddev)
      in
      let scaled_diff = (x -$ mean) /$ stddev in
      let squared_error = scaled_diff *$ scaled_diff /$. 2. in
      let regulariser = (Float.log (2. *. Float.pi) /. 2.) +.$ ln stddev in
      reduction @@ (~-$squared_error +$ regulariser)
  | Uniform (low, high) ->
      let low, high =
        match batch_size with
        | Some batch_size ->
            let low = Var.BroadcastInDim (low, [batch_size]) in
            let high = Var.BroadcastInDim (high, [batch_size]) in
            (low, high)
        | None ->
            (low, high)
      in
      reduction @@ ~-$(ln (high -$ low))
  | Beta (alpha, beta) ->
      let alpha, beta =
        match batch_size with
        | Some batch_size ->
            let alpha = Var.BroadcastInDim (alpha, [batch_size]) in
            let beta = Var.BroadcastInDim (beta, [batch_size]) in
            (alpha, beta)
        | None ->
            (alpha, beta)
      in
      reduction @@ ((alpha -$. 1.) *$ ln x)
         +$ ((beta -$. 1.) *$ ln (1. -.$ x))
         -$ ln_beta alpha beta
  | Exponential rate ->
      let rate =
        match batch_size with
        | Some batch_size ->
            Var.BroadcastInDim (rate, [batch_size])
        | None ->
            rate
      in
      reduction @@ (ln rate -$ (rate *$ x))

let expectation dist =
  match dist with
  | Normal (mean, _) ->
      mean
  | Uniform (low, high) ->
      (low +$ high) /$. 2.
  | Beta (alpha, beta) ->
      alpha /$ (alpha +$ beta)
  | Exponential rate ->
      1. /.$ rate

let kl ?(reduction=(fun x -> sum x)) p q =
  (* p = guide q = prior *)
  match (p, q) with
  | Normal (mean_p, std_p), Normal (mean_q, std_q) ->
      let std_ratio = std_p /$ std_q in
      let var_ratio = std_ratio *$ std_ratio in
      let scaled_diff = (mean_p -$ mean_q) /$ std_q in
      let squared_diff = scaled_diff *$ scaled_diff in
      Some (reduction ((var_ratio +$ squared_diff -$. 1. -$ ln var_ratio) /$. 2.))
  | Beta (alpha_p, beta_p), Beta (alpha_q, beta_q) ->
      let alpha_sum_p = alpha_p +$ beta_p in
      Some
        (reduction
        @@ ln_beta alpha_q beta_q -$ ln_beta alpha_p beta_p
           +$ ((alpha_p -$ alpha_q) *$ digamma alpha_p)
           +$ ((beta_p -$ beta_q) *$ digamma beta_p)
           +$ ((alpha_q -$ alpha_p +$ beta_q -$ beta_p) *$ digamma alpha_sum_p)
        )
  | Exponential rate_p, Exponential rate_q ->
      Some (reduction @@ (ln (rate_p /$ rate_q) +$ (rate_q /$ rate_p) -$. 1.))
  | Uniform (low_p, high_p), Uniform (low_q, high_q) ->
      Some (reduction @@ ln ((high_q -$ low_q) /$ (high_p -$ low_p)))
  | Normal (mean_p, std_p), Uniform (low_q, high_q) ->
      let range = high_q -$ low_q in
      let sqrt_two_pi = Float.sqrt (2.0 *. Float.pi) *.$ ones_like mean_p in
      let term1 = ln (range *$ sqrt_two_pi *$ std_p) in
      let two = 2.0 *.$ ones_like mean_p in
      let term2 =
        (mean_p -$ low_q) *$ (mean_p -$ low_q) /$ (two *$ std_p *$ std_p)
      in
      let term3 =
        (mean_p -$ high_q) *$ (mean_p -$ high_q) /$ (two *$ std_p *$ std_p)
      in
      Some (reduction @@ (term1 +$ term2 +$ term3))
  | Beta (alpha_p, beta_p), Normal (mean_q, std_q) ->
      let sqrt_two_pi = Float.sqrt (2.0 *. Float.pi) *.$ ones_like alpha_p in
      let term1 = ln (sqrt_two_pi *$ std_q) in
      let two = 2.0 *.$ ones_like alpha_p in
      let term2 =
        ((alpha_p /$ (alpha_p +$ beta_p)) -$ mean_q)
        *$ ((alpha_p /$ (alpha_p +$ beta_p)) -$ mean_q)
        /$ (two *$ std_q *$ std_q)
      in
      let term3 = ln_beta alpha_p beta_p in
      Some (reduction @@ (term1 +$ term2 +$ term3))
  | Exponential rate_p, Normal (mean_q, std_q) ->
      let sqrt_two_pi = Float.sqrt (2.0 *. Float.pi) *.$ ones_like rate_p in
      let term1 = ln (sqrt_two_pi *$ std_q) in
      let two = 2.0 *.$ ones_like rate_p in
      let one = 1.0 *.$ ones_like rate_p in
      let term2 =
        ((one /$ rate_p) -$ mean_q)
        *$ ((one /$ rate_p) -$ mean_q)
        /$ (two *$ std_q *$ std_q)
      in
      let term3 = ln rate_p in
      Some (reduction @@ (term1 +$ term2 +$ term3))
  | _ ->
      (* if there is no closed form solution or good approximation, we just use monte carlo *)
      None

let to_string dist =
  match dist with
  | Normal (mean, std) ->
      Printf.sprintf "Normal(mean=%s, std=%s)" (Var.to_string mean) (Var.to_string std)
  | Uniform (low, high) ->
      Printf.sprintf "Uniform(low=%s, high=%s)" (Var.to_string low) (Var.to_string high)
  | Beta (alpha, beta) ->
      Printf.sprintf "Beta(alpha=%s, beta=%s)" (Var.to_string alpha) (Var.to_string beta)
  | Exponential rate ->
      Printf.sprintf "Exponential(rate=%s)" (Var.to_string rate)
