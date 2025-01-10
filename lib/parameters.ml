type ('a, 'b, 'c) inner =
  { output_type: 'c Ir.ValueType.t
  ; old_params: 'b Hlist.hlist Ir.Var.t -> 'a Hlist.hlist Ir.Var.t
  ; initial_values: 'b Hlist.hlist Runtime.HostValue.t
  ; f: 'b Hlist.hlist Ir.Var.t -> 'c Ir.Var.t }

type ('a, 'b, 'c) t = 'a Hlist.hlist Runtime.HostValue.t -> ('a, 'b, 'c) inner

let return : type a b. a Ir.Var.t -> (b, b, a) t =
 fun x initial_values ->
  { old_params= Fun.id
  ; output_type= Ir.ValueType.of_vars x
  ; initial_values
  ; f= (fun _ -> x) }

let bind :
    type a b c d e. (a, b, c) t -> (c Ir.Var.t -> (b, d, e) t) -> (a, d, e) t =
 fun x f initial_values ->
  let dummy_output = f (Ir.ValueType.to_arg (x initial_values).output_type) in
  let dummy_inner = dummy_output (x initial_values).initial_values in
  { dummy_inner with
    f=
      (fun z ->
        let b_param = dummy_inner.old_params z in
        let x_inner = x initial_values in
        let c = x_inner.f b_param in
        let output = f c in
        let output_inner = output x_inner.initial_values in
        output_inner.f z )
  ; old_params=
      (fun z ->
        let b_param = dummy_inner.old_params z in
        (x initial_values).old_params b_param ) }

let ( let* ) = bind

let new_param : type a b. a Runtime.HostValue.t -> (b, a -> b, a) t =
 fun t xs ->
  { initial_values= t :: xs
  ; old_params= (fun (_ :: xs) -> xs)
  ; output_type= Runtime.HostValue.value_type t
  ; f= (fun (x :: _) -> x) }

let apply : type a b. (unit, a, b) t -> a Hlist.hlist Ir.Var.t -> b Ir.Var.t =
 fun x -> (x []).f

let initial :
    type a b c.
       a Ir.ValueType.t
    -> (a Ir.Var.t -> (unit, b, c) t)
    -> b Hlist.hlist Runtime.HostValue.t =
 fun t f ->
  let dummy_x = Ir.ValueType.to_arg t in
  let dummy_inner = Random.dummy_handler (fun () -> f dummy_x []) in
  dummy_inner.initial_values

let param_type :
    type a b c.
       a Ir.ValueType.t
    -> (a Ir.Var.t -> (unit, b, c) t)
    -> b Hlist.hlist Ir.ValueType.t =
 fun t f -> initial t f |> Runtime.HostValue.value_type

let grad_and_value :
    type a b c.
    (a, b, c) t -> (a, b, (b Hlist.hlist -> c -> unit) Hlist.hlist) t =
 fun x initial_values ->
  let x = x initial_values in
  { x with
    output_type= [Runtime.HostValue.value_type x.initial_values; x.output_type]
  ; f= Backpropagate.diff Var x.f }

let params : type a. (a, a, a Hlist.hlist) t =
 fun initial_values ->
  { initial_values
  ; old_params= Fun.id
  ; output_type= Runtime.HostValue.value_type initial_values
  ; f= Fun.id }

let create_func t f =
  let param_type = param_type t f in
  let input_types = Ir.ValueType.List.[param_type; t; E Random.seed_type] in
  Ir.create_func input_types (fun [params; x; E seed] ->
      Random.handler
        (fun () ->
          let y = apply (f x) params in
          let seed = Random.current_seed () in
          Ir.Var.List.[y; E seed] )
        seed )
