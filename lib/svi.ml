type _ Effect.t +=
  | Sample :
      Distribution.t * Distribution.t
      -> (Ir.Tensor.f32, float) Ir.Var.u Effect.t

let sample ~prior ~guide = Effect.perform (Sample (prior, guide))

let elbo observation f =
  let open Effect.Deep in
  let open Dsl in
  let elbo = ref ~.0. in
  let distribution =
    try_with f ()
      { effc=
          (fun (type a) (eff : a Effect.t) ->
            match eff with
            | Sample (prior, guide) ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let sample = Distribution.sample guide in
                    let elbo' =
                      Distribution.log_prob prior sample
                      -@ Distribution.log_prob guide sample
                    in
                    elbo := !elbo +@ elbo' ;
                    continue k sample )
            | _ ->
                None ) }
  in
  !elbo +@ Distribution.log_prob distribution observation
