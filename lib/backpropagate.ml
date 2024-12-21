(* (input_type, old_output_type, list end, output list) *)
open Dsl

type (_, _, _, _) input =
  | Var : ('a, 'b, 'c, 'a Ir.Var.t -> 'c) input
  | Const : ('a, 'b, 'c, 'c) input
  | [] : (unit Ir.VarList.t, 'a, 'b, unit Ir.VarList.t Ir.Var.t -> 'b) input
  | ( :: ) :
      ('a, 'b, 'c, 'd) input
      * ('e Ir.VarList.t, 'b, 'f, 'c Ir.VarList.t Ir.Var.t -> 'f) input
      -> ( ('a Ir.Var.t -> 'e) Ir.VarList.t
         , 'b
         , 'f
         , 'd Ir.VarList.t Ir.Var.t -> 'f )
         input

let diff :
    type a b c.
       (a, b, b Ir.Var.t -> unit, c) input
    -> (a Ir.Var.t -> b Ir.Var.t)
    -> a Ir.Var.t
    -> c Ir.VarList.t Ir.Var.t =
 fun l f inputs ->
  let opt_add :
      type a.
         a Ir.tensor Ir.Var.t option
      -> a Ir.tensor Ir.Var.t option
      -> a Ir.tensor Ir.Var.t option =
   fun x y ->
    match (x, y) with
    | Some x, Some y ->
        Some (x +@ y)
    | Some x, None | None, Some x ->
        Some x
    | None, None ->
        None
  in
  let opt_sub :
      type a.
         a Ir.tensor Ir.Var.t option
      -> a Ir.tensor Ir.Var.t option
      -> a Ir.tensor Ir.Var.t option =
   fun x y ->
    match (x, y) with
    | Some x, Some y ->
        Some (x -@ y)
    | Some x, None ->
        Some x
    | None, Some x ->
        Some (zeros_like x -@ x)
    | None, None ->
        None
  in
  let rec backprop :
      type a. a Ir.Var.t -> a Ir.Var.t -> int -> a Ir.Var.t option =
   fun v grad x ->
    match v with
    | Ir.Var.Add (v1, v2) ->
        opt_add (backprop v1 grad x) (backprop v2 grad x)
    | Subtract (v1, v2) ->
        opt_sub (backprop v1 grad x) (backprop v2 grad x)
    | Multiply (v1, v2) ->
        opt_add (backprop v2 (v1 *@ grad) x) (backprop v1 (v2 *@ grad) x)
    | Abs v ->
        Option.map abs (backprop v grad x) (* TODO: This is wrong *)
    | Ln v ->
        backprop v (grad /@ v) x
    | Exponential v ->
        backprop v (grad *@ Exponential v) x
    | Pow (v1, v2) ->
        opt_add
          (backprop v1 (grad *@ v *@ v1 /@ v2) x)
          (backprop v2 (grad *@ v *@ ln v1) x)
    | Argument _ ->
        None
    | Compare _ ->
        failwith "cannot differentiate binary comparison"
    | Min _ ->
        failwith "cannot differentiate min"
    | Max _ ->
        failwith "cannot differentiate max"
    | Constant _ ->
        None
    | DotProduct
        ( lhs
        , rhs
        , lhs_contracting_dims
        , rhs_contracting_dims
        , lhs_batching_dims
        , rhs_batching_dims ) ->
        let backprop_dot first var var_contracting_dims var_batching_dims const
            const_contracting_dims const_batching_dims =
          let var_shape = Ir.shape_of_var var in
          let var_rem_dims =
            List.init (List.length var_shape) Fun.id
            |> List.filter (fun i ->
                   not
                     ( List.mem i var_batching_dims
                     || List.mem i var_contracting_dims ) )
          in
          let permutation =
            List.mapi
              (fun i _ ->
                match List.find_index (( = ) i) var_batching_dims with
                | Some j ->
                    j
                | None -> (
                  match List.find_index (( = ) i) var_contracting_dims with
                  | Some j ->
                      j + List.length var_batching_dims
                  | None ->
                      List.length var_batching_dims
                      + List.length var_contracting_dims
                      + (List.find_index (( = ) i) var_rem_dims |> Option.get) )
                )
              var_shape
          in
          let grad_shape = Ir.shape_of_var grad in
          let const_shape = Ir.shape_of_var const in
          let grad_dims = List.init (List.length grad_shape) Fun.id in
          let grad_contracting_dims =
            List.filter
              (fun i ->
                if first then
                  i + List.length const_shape
                  >= List.length grad_shape
                     + List.length const_contracting_dims
                     + List.length const_batching_dims
                else
                  i >= List.length const_batching_dims
                  && i + List.length const_contracting_dims
                     < List.length const_shape )
              grad_dims
          in
          let grad_batching_dims =
            List.filter (fun i -> i < List.length var_batching_dims) grad_dims
          in
          let const_dims = List.init (List.length const_shape) Fun.id in
          let const_contracting_dims =
            List.filter
              (fun i ->
                (not (List.mem i const_contracting_dims))
                && not (List.mem i const_batching_dims) )
              const_dims
          in
          let prod =
            Ir.Var.DotProduct
              ( const
              , grad
              , const_contracting_dims
              , grad_contracting_dims
              , const_batching_dims
              , grad_batching_dims )
          in
          Ir.Var.Transpose (prod, permutation)
        in
        opt_add
          (backprop lhs
             (backprop_dot true lhs lhs_contracting_dims lhs_batching_dims rhs
                rhs_contracting_dims rhs_batching_dims )
             x )
          (backprop rhs
             (backprop_dot false rhs rhs_contracting_dims rhs_batching_dims lhs
                lhs_contracting_dims lhs_batching_dims )
             x )
    | Random _ ->
        None
    | [] | _ :: _ ->
        failwith "lists cannot be backpropagated"
    | DiffVar (id, _) ->
        if Int.equal id x then Some grad else None
    | DiffConst _ ->
        None
    | Divide (v1, v2) ->
        opt_add
          (backprop v1 (grad /@ v2) x)
          (backprop v2 (grad *@ (Dsl.zeros_like v1 -@ v1) /@ (v2 *@ v2)) x)
    | BroadcastInDim (var, dims) -> (
      match Ir.ValueType.of_var var with
      | Ir.ValueType.Tensor_type (_, Ir.F32) ->
          let reduced_grad =
            Dsl.sum (List.init (List.length dims) Fun.id) grad
          in
          backprop var reduced_grad x
      | _ ->
          failwith "broadcasting only supported for f32 tensors" )
    | Transpose (var, permutation) ->
        let inverse_permutation =
          List.init (List.length permutation) (fun i ->
              List.find_index (fun x -> x = i) permutation |> Option.get )
        in
        backprop var (Ir.Var.Transpose (grad, inverse_permutation)) x
    | Tanh var ->
        backprop var (grad *@ (Dsl.ones_like var -@ (tanh var *@ tanh var))) x
        (* Jax uses different backpropagation that is more numerically precise, but also slower *)
    | Sum (var, dimensions) ->
        let var_shape = Ir.shape_of_var var in
        let new_dims =
          List.filteri (fun i _ -> List.mem i dimensions) var_shape
        in
        let new_grad = Ir.Var.BroadcastInDim (grad, new_dims) in
        let new_dims =
          List.init (List.length var_shape) Fun.id
          |> List.filteri (fun i _ -> List.mem i dimensions)
        in
        let old_dims =
          List.init (List.length var_shape) Fun.id
          |> List.filteri (fun i _ -> not (List.mem i dimensions))
        in
        let perm =
          List.init (List.length var_shape) Fun.id
          |> List.map (fun i ->
                 match List.find_index (( = ) i) new_dims with
                 | Some j ->
                     j
                 | None ->
                     (List.find_index (( = ) i) old_dims |> Option.get)
                     + List.length new_dims )
        in
        let new_grad = Ir.Var.Transpose (new_grad, perm) in
        backprop var new_grad x
  in
  let rec wrap_inputs :
      type a b c d. (a, b, c, d) input -> a Ir.Var.t -> a Ir.Var.t =
   fun l1 l2 ->
    match (l1, l2) with
    | [], [] ->
        []
    | x :: xs, y :: ys ->
        let input = wrap_inputs x y in
        let inputs = wrap_inputs xs ys in
        input :: inputs
    | Var, x ->
        Ir.Var.map
          { fn=
              (fun x ->
                let id = Ir.new_id () in
                DiffVar (id, x) ) }
          x
    | Const, x ->
        Ir.Var.map {fn= (fun x -> DiffConst x)} x
  in
  let rec initial_grad : type a. a Ir.ValueType.t -> a Ir.Var.t = function
    | Ir.ValueType.Tensor_type _ as t ->
        Dsl.ones t
    | List_type t ->
        let open Hlist.Map (Ir.ValueTypeList) (Ir.VarList) in
        map {f= initial_grad} t |> Ir.Var.from_var_list
  in
  let assert_same_type : type a b. a Ir.Var.t -> b Ir.Var.t -> b Ir.Var.t =
   fun x y ->
    match (Ir.ValueType.of_var x, Ir.ValueType.of_var y) with
    | ( Ir.ValueType.Tensor_type (s1, Ir.F32)
      , Ir.ValueType.Tensor_type (s2, Ir.F32) )
      when s1 = s2 ->
        x
    | Ir.ValueType.Tensor_type (s1, Ir.I1), Ir.ValueType.Tensor_type (s2, Ir.I1)
      when s1 = s2 ->
        x
    | ( Ir.ValueType.Tensor_type (s1, Ir.I64)
      , Ir.ValueType.Tensor_type (s2, Ir.I64) )
      when s1 = s2 ->
        x
    | Ir.ValueType.Tensor_type _, Ir.ValueType.Tensor_type _ ->
        failwith "different tensor types"
        (* TODO: Make this less hacky. Maybe just cast? *)
    | _ ->
        failwith "different types"
  in
  let rec iter_vars :
      type a b c d.
         (a, b, c, d) input
      -> a Ir.Var.t
      -> b Ir.Var.t
      -> c Ir.VarList.t Ir.Var.t
      -> d Ir.VarList.t Ir.Var.t =
   fun l inputs outputs next ->
    match (l, inputs) with
    | [], [] ->
        [] :: next
    | x :: xs, y :: ys ->
        let (outputs' :: _) = iter_vars xs ys outputs next in
        let output = iter_vars x y outputs outputs' in
        output :: next
    | Var, x :: xs ->
        let (outputs' :: _) = iter_vars Var xs outputs next in
        let output = iter_vars Var x outputs outputs' in
        output :: next
    | Var, [] ->
        [] :: next
    | Var, DiffVar (id, x) ->
        let initial_grad = initial_grad (Ir.ValueType.of_var outputs) in
        let output = backprop outputs initial_grad id in
        let output =
          match output with
          | Some x ->
              x
          | None ->
              failwith "output does not depend on input"
        in
        let output = assert_same_type output x in
        output :: next
    | Const, DiffConst _ ->
        next
    | _ ->
        failwith "should be impossible"
  in
  let inputs = wrap_inputs l inputs in
  let outputs = f inputs in
  iter_vars l inputs outputs [outputs]
