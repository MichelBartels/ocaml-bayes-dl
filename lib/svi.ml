let sample ~prior ~guide = Effect.perform (Distribution.Sample (prior, guide))

let elbo observation parametrised_distr =
  let open Parameters in
  flatten
  @@
  let open Effect.Deep in
  let* params = params_for parametrised_distr in
  let f = to_fun parametrised_distr in
  let open Dsl in
  let kl = ref ~.0. in
  let distribution =
    try_with f params
      { effc=
          (fun (type a) (eff : a Effect.t) ->
            match eff with
            | Distribution.Sample (prior, guide) ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let sample = Distribution.sample guide in
                    let kl' = Distribution.kl guide prior in
                    kl := !kl +@ kl' ;
                    continue k sample )
            | _ ->
                None ) }
  in
  return @@ Ir.Var.List.E (!kl -@ Distribution.log_prob distribution observation)

let inference parametrised =
  let open Parameters in
  flatten
  @@
  let open Effect.Deep in
  let* params = params_for parametrised in
  let f = to_fun parametrised in
  return
  @@ try_with f params
       { effc=
           (fun (type a) (eff : a Effect.t) ->
             match eff with
             | Distribution.Sample (_, guide) ->
                 Some
                   (fun (k : (a, _) continuation) ->
                     let sample = Distribution.sample guide in
                     continue k sample )
             | _ ->
                 None ) }
