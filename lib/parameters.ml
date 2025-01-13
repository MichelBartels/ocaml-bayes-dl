type ('a, 'b) t = ('a Ir.Var.t -> 'b Ir.Var.t) * 'a Runtime.HostValue.t

let return : type a. a Ir.Var.t -> (unit Hlist.hlist, a) t =
 fun x -> ((fun [] -> x), [])

let bind : type a b c d.
       (a, b) t
    -> (b Ir.Var.t -> (c Hlist.hlist, d) t)
    -> ((a -> c) Hlist.hlist, d) t =
 fun (f, params) f' ->
  ( (fun (x :: xs) ->
      let y = f x in
      let f', _ = f' y in
      f' xs )
  , let dummy_params =
      Ir.ValueType.to_arg @@ Runtime.HostValue.value_type params
    in
    let dummy_output = f dummy_params in
    let _, paramss = f' dummy_output in
    params :: paramss )

let ( let* ) = bind

let new_param : type a. a Runtime.HostValue.t -> (a, a) t =
 fun t -> ((fun x -> x), t)

let to_fun : type a b. (a, b) t -> a Ir.Var.t -> b Ir.Var.t = fst

let initial : type a b c.
    a Ir.ValueType.t -> (a Ir.Var.t -> (b, c) t) -> b Runtime.HostValue.t =
 fun t f ->
  let dummy_x = Ir.ValueType.to_arg t in
  Random.dummy_handler (fun () -> snd @@ f dummy_x)

let params_for : type a b. (a, b) t -> (a, a) t =
 fun (_, params) -> (Fun.id, params)

let param_type t f = Runtime.HostValue.value_type @@ initial t f
