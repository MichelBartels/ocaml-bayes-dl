type _ Effect.t +=
  | Counter : int -> (Ir.Tensor.u64, Unsigned.uint64) Ir.Var.u Effect.t
  | Sample :
      Distribution.t * Distribution.t * int option
      -> (Ir.Tensor.f32, float) Ir.Var.u Effect.t

let dummy_handler f =
  let open Effect.Deep in
  try_with f ()
    { effc=
        (fun (type a) (eff : a Effect.t) ->
          match eff with
          | Counter _ ->
              Some
                (fun (k : (a, _) continuation) ->
                  continue k @@ Dsl.scalar_u64 "0" )
          | Sample (_, d, batch_size) ->
              Some
                (fun (k : (a, _) continuation) ->
                  continue k @@ Distribution.sample d batch_size )
          | _ ->
              None ) }
