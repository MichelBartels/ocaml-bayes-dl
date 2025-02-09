type ('a, 'b) t = ('a Ir.Var.t -> 'b) * 'a Runtime.HostValue.t

let return : type a. a -> (unit Hlist.hlist, a) t = fun x -> ((fun [] -> x), [])

let bind : type a b c d.
    (a, b) t -> (b -> (c Hlist.hlist, d) t) -> ((a -> c) Hlist.hlist, d) t =
 fun (f, params) f' ->
  ( (fun (x :: xs) ->
      let y = f x in
      let f', _ = f' y in
      f' xs )
  , let dummy_params =
      Ir.ValueType.to_arg @@ Runtime.HostValue.value_type params
    in
    Effects.dummy_handler
    @@ fun () ->
    let dummy_output = f dummy_params in
    let _, paramss = f' dummy_output in
    Runtime.HostValue.(params :: paramss) )

let ( let* ) = bind

let new_param : type a. a Runtime.HostValue.t -> (a, a Ir.Var.t) t =
 fun t -> ((fun x -> x), t)

let to_fun : type a b. (a, b) t -> a Ir.Var.t -> b = fst

let initial : type a b c.
    a Ir.ValueType.t -> (a Ir.Var.t -> (b, c) t) -> b Runtime.HostValue.t =
 fun t f ->
  let dummy_x = Ir.ValueType.to_arg t in
  Effects.dummy_handler @@ fun () -> snd @@ f dummy_x

let params_for : type a b. (a, b) t -> (a, a Ir.Var.t) t =
 fun (_, params) -> (Fun.id, params)

let param_type t f = Runtime.HostValue.value_type @@ initial t f

let flatten : type a b. ((a -> unit) Hlist.hlist, b) t -> (a, b) t =
 fun (f, [params]) -> ((fun x -> f [x]), params)
