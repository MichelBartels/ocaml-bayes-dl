let sample ~prior ~guide ?batch_size () =
  Effect.perform (Distribution.Sample (prior, guide, batch_size))

let elbo ?(only_kl = false) observation parametrised_distr =
  let open Parameters in
  flatten
  @@
  let open Effect.Deep in
  let* params = params_for parametrised_distr in
  let f = to_fun parametrised_distr in
  let open Dsl in
  let kls = ref [] in
  let distribution =
    try_with f params
      { effc=
          (fun (type a) (eff : a Effect.t) ->
            match eff with
            | Distribution.Sample (prior, guide, batch_size) ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let sample = Distribution.sample guide batch_size in
                    let kl = match Distribution.kl guide prior with
                      | Some kl -> kl
                      | None ->
                        let log_prob_guide = Distribution.log_prob ?batch_size guide sample in
                        let log_prob_prior = Distribution.log_prob ?batch_size prior sample in
                        log_prob_guide -$ log_prob_prior
                    in
                    kls := kl :: !kls ;
                    continue k sample )
            | _ ->
                None ) }
  in
  let kl = List.fold_left ( +$ ) ~.0. !kls in
  return
  @@ Var.List.E
       ( kl
       -$ Distribution.log_prob distribution observation
          *$. if only_kl then 0. else 1. )

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
             | Distribution.Sample (_, guide, batch_size) ->
                 Some
                   (fun (k : (a, _) continuation) ->
                     let sample = Distribution.sample guide batch_size in
                     continue k sample )
             | _ ->
                 None ) }
