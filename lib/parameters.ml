type ('a, 'b, 'c) inner =
  { new_param_types: 'b Ir.VarList.t Ir.ValueType.t
  ; output_type: 'c Ir.ValueType.t
  ; old_params: 'b Ir.VarList.t Ir.Var.t -> 'a Ir.VarList.t Ir.Var.t
  ; f: 'b Ir.VarList.t Ir.Var.t -> 'c Ir.Var.t }

type ('a, 'b, 'c) t = 'a Ir.VarList.t Ir.ValueType.t -> ('a, 'b, 'c) inner

let return : type a b. a Ir.Var.t -> (b, b, a) t =
 fun x arg_types ->
  { new_param_types= arg_types
  ; old_params= Fun.id
  ; output_type= Ir.ValueType.of_var x
  ; f= (fun _ -> x) }

let bind :
    type a b c d e. (a, b, c) t -> (c Ir.Var.t -> (b, d, e) t) -> (a, d, e) t =
 fun x f arg_types ->
  let dummy_output = f (Ir.ValueType.to_arg (x arg_types).output_type) in
  let dummy_inner = dummy_output (x arg_types).new_param_types in
  { dummy_inner with
    f=
      (fun z ->
        let b_param = dummy_inner.old_params z in
        let a_param = (x arg_types).old_params b_param in
        let x_inner = x @@ Ir.ValueType.of_var a_param in
        let c = x_inner.f b_param in
        let output = f c in
        let output_inner = output @@ Ir.ValueType.of_var b_param in
        output_inner.f z )
  ; old_params=
      (fun z ->
        let b_param = dummy_inner.old_params z in
        (x arg_types).old_params b_param ) }

let ( let* ) = bind

let new_param : type a b. a Ir.ValueType.t -> (b, a Ir.Var.t -> b, a) t =
 fun t (List_type xs) ->
  { new_param_types= List_type (t :: xs)
  ; old_params= (fun (_ :: xs) -> xs)
  ; output_type= t
  ; f= (fun (x :: _) -> x) }

let to_fun : type a b. (unit, a, b) t -> a Ir.VarList.t Ir.Var.t -> b Ir.Var.t =
 fun x -> (x (List_type [])).f

let grad_and_value :
    type a b c.
       (a, b, c) t
    -> (a, b, (b Ir.VarList.t Ir.Var.t -> c Ir.Var.t -> unit) Ir.VarList.t) t =
 fun x arg_type ->
  let x = x arg_type in
  { x with
    output_type= List_type [x.new_param_types; x.output_type]
  ; f= Backpropagate.diff Var x.f }

let params : type a. (a, a, a Ir.VarList.t) t =
 fun arg_type ->
  { new_param_types= arg_type
  ; old_params= Fun.id
  ; output_type= arg_type
  ; f= Fun.id }

let create_func t f =
  let dummy_x = Ir.ValueType.to_arg t in
  let dummy_inner =
    Random.dummy_handler (fun () -> f dummy_x (Ir.ValueType.List_type []))
  in
  let input_types =
    Ir.ValueType.List_type [dummy_inner.new_param_types; t; Random.seed_type]
  in
  Ir.create_func input_types (fun [params; x; seed] ->
      Random.handler
        (fun () ->
          let inner = f x (List_type []) in
          let y = inner.f params in
          let seed = Random.current_seed () in
          Ir.Var.[y; seed] )
        seed )
