open Dsl

(* (input_type, old_output_type, list end, output list) *)
type (_, _, _, _) input =
  | Var : ('a, 'b, 'c, 'a -> 'c) input
  | Const : ('a, 'b, 'c, 'c) input
  | [] : (unit Hlist.hlist, 'a, 'b, unit Hlist.hlist -> 'b) input
  | ( :: ) :
      ('a, 'b, 'c, 'd) input
      * ('e Hlist.hlist, 'b, 'f, 'c Hlist.hlist -> 'f) input
      -> (('a -> 'e) Hlist.hlist, 'b, 'f, 'd Hlist.hlist -> 'f) input

module VarSet = Set.Make (struct
  type t = Var.any

  let compare = Stdlib.compare
end)

let topological_order : type a b. (a, b) Var.u -> Var.any list =
 fun var ->
  let rec loop : type a b.
      VarSet.t -> Var.any list -> (a, b) Var.u -> VarSet.t * Var.any list =
   fun visited order var ->
    if VarSet.mem (Any var) visited then (visited, order)
    else
      let visited = VarSet.add (Any var) visited in
      match var with
      | Add (x, y)
      | Subtract (x, y)
      | Multiply (x, y)
      | Divide (x, y)
      | Pow (x, y)
      | DotProduct (x, y, _, _, _, _) ->
          let visited, order = loop visited order x in
          let visited, order = loop visited order y in
          let var = Var.Any var in
          (visited, var :: order)
      | Abs x
      | Ln x
      | Exponential x
      | BroadcastInDim (x, _)
      | Transpose (x, _)
      | Reshape (x, _)
      | Sin x
      | Cos x
      | Tanh x ->
          let visited, order = loop visited order x in
          let var = Var.Any var in
          (visited, var :: order)
      | Argument _
      | Constant _
      | BroadcastScalarConstant _
      | Random _
      | DiffVar _
      | RightShift _
      | LeftShift _
      | NoGrad _ ->
          (visited, order)
      | Compare _ | Min _ | Max _ | Or _ | Iota _ | Bitcast _ | Convert _ ->
          failwith "not differentiable"
      | Sum (x, _) ->
          let visited, order = loop visited order x in
          let var = Var.Any var in
          (visited, var :: order)
      | Concatenate (xs, _) ->
          let visited, order =
            List.fold_left
              (fun (visited, order) -> loop visited order)
              (visited, order) xs
          in
          let var = Var.Any var in
          (visited, var :: order)
  in
  loop VarSet.empty [] var |> snd

let assert_same_type' : type a b c d.
    (a, b) Ir.Var.u -> (c, d) Ir.Var.u -> (c, d) Ir.Var.u =
 fun x y ->
  match (Ir.ValueType.of_var x, Ir.ValueType.of_var y) with
  | (s1, F32), (s2, F32) when s1 = s2 ->
      x
  | (_, F32), (_, F32) ->
      failwith "different float tensor shapes"
  | (s1, I1), (s2, I1) when s1 = s2 ->
      x
  | (s1, I64), (s2, I64) when s1 = s2 ->
      x
  | _, _ ->
      failwith "different tensor types"
(* TODO: Make this less hacky. Maybe just cast? *)

let rec assert_same_type : type a b. a Ir.Var.t -> b Ir.Var.t -> b Ir.Var.t =
 fun x y ->
  match (x, y) with
  | [], [] ->
      []
  | x :: xs, y :: ys ->
      let x = assert_same_type x y in
      x :: assert_same_type xs ys
  | E x, E y ->
      E (assert_same_type' x y)
  | _ ->
      failwith "different type"

module GradMap = struct
  module VarMap = Map.Make (struct
    type t = Var.any

    let compare = Stdlib.compare
  end)

  type t = Var.any VarMap.t

  let empty = VarMap.empty

  let add : type a b. t -> (a, b) Var.u -> (a, b) Var.u -> t =
   fun map var grad ->
    VarMap.update (Any var)
      (function
        | None ->
            Some (Var.Any grad)
        | Some (Var.Any grad') ->
            let grad' = assert_same_type' grad' grad in
            Some (Any (grad +@ grad')) )
      map

  let get : type a b. t -> (a, b) Var.u -> (a, b) Var.u option =
   fun map var ->
    match VarMap.find_opt (Any var) map with
    | Some (Any grad) ->
        let grad = assert_same_type' grad var in
        Some grad
    | None ->
        None
end

let diff : type a b c d.
       (a, (b, d) Hlist.element, (b, d) Hlist.element -> unit, c) input
    -> (a Ir.Var.t -> (b, d) Hlist.element Ir.Var.t)
    -> a Ir.Var.t
    -> c Hlist.hlist Ir.Var.t =
 fun l f inputs ->
  let rec wrap_inputs : type a b c d.
      (a, b, c, d) input -> a Ir.Var.t -> a Ir.Var.t =
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
          { f=
              (fun x ->
                let id = Ir.new_id () in
                DiffVar (id, x) ) }
          x
    | Const, x ->
        Ir.Var.map {f= (fun x -> NoGrad x)} x
  in
  let inputs = wrap_inputs l inputs in
  let (E output) = f inputs in
  let order = topological_order output in
  let backprop : type a b. (a, b) Ir.Var.u -> GradMap.t -> GradMap.t =
   fun var grads ->
    match GradMap.get grads var with
    | None ->
        grads
    | Some grad -> (
      match var with
      | Ir.Var.Add (v1, v2) ->
          let grads = GradMap.add grads v1 grad in
          GradMap.add grads v2 grad
      | Subtract (v1, v2) ->
          let grads = GradMap.add grads v1 grad in
          GradMap.add grads v2 (ones_like grad -@ grad)
      | Multiply (v1, v2) ->
          let grads = GradMap.add grads v1 (grad *@ v2) in
          GradMap.add grads v2 (grad *@ v1)
      | Abs _ ->
          failwith "abs not yet implemented"
      | Ln v ->
          GradMap.add grads v (grad /@ v)
      | Exponential v ->
          GradMap.add grads v (grad *@ Exponential v)
      | Pow (v1, v2) ->
          let grads = GradMap.add grads v1 (grad *@ var *@ v2 /@ v1) in
          GradMap.add grads v2 (grad *@ ln v1 *@ var)
      | Argument _
      | Compare _
      | Min _
      | Max _
      | Constant _
      | BroadcastScalarConstant _
      | Random _
      | DiffVar _
      | RightShift _
      | LeftShift _
      | Bitcast _
      | Convert _
      | NoGrad _
      | Or _
      | Iota _ ->
          failwith "should not be reachable"
      | DotProduct
          ( lhs
          , rhs
          , lhs_contracting_dims
          , rhs_contracting_dims
          , lhs_batching_dims
          , rhs_batching_dims ) ->
          let backprop_dot first var var_contracting_dims var_batching_dims
              const const_contracting_dims const_batching_dims =
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
                        + (List.find_index (( = ) i) var_rem_dims |> Option.get)
                    ) )
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
          let grads =
            GradMap.add grads lhs
              (backprop_dot true lhs lhs_contracting_dims lhs_batching_dims rhs
                 rhs_contracting_dims rhs_batching_dims )
          in
          GradMap.add grads rhs
            (backprop_dot false rhs rhs_contracting_dims rhs_batching_dims lhs
               lhs_contracting_dims lhs_batching_dims )
      | Divide (v1, v2) ->
          let grads = GradMap.add grads v1 (grad /@ v2) in
          GradMap.add grads v2 (grad *@ (Dsl.zeros_like v1 -@ v1) /@ (v2 *@ v2))
      | BroadcastInDim (var, dims) -> (
        match Ir.ValueType.of_var var with
        | _, Ir.Tensor.F32 ->
            let reduced_grad =
              Dsl.sum (List.init (List.length dims) Fun.id) grad
            in
            GradMap.add grads var reduced_grad
        | _ ->
            failwith "can only differentiate broadcasting of f32 tensors" )
      | Transpose (var, permutation) ->
          let inverse_permutation =
            List.init (List.length permutation) (fun i ->
                List.find_index (fun x -> x = i) permutation |> Option.get )
          in
          let grad = Ir.Var.Transpose (grad, inverse_permutation) in
          GradMap.add grads var grad
      | Tanh var ->
          GradMap.add grads var
            (grad *@ (Dsl.ones_like var -@ (tanh var *@ tanh var)))
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
          let grad = Ir.Var.Transpose (new_grad, perm) in
          GradMap.add grads var grad
      | Reshape (var, _) ->
          let grad = reshape (Ir.shape_of_var var) grad in
          GradMap.add grads var grad
      | Sin var ->
          GradMap.add grads var (grad *@ cos var)
      | Cos var ->
          GradMap.add grads var (grad *@ (Dsl.zeros_like var -@ sin var))
      | Concatenate _ ->
          failwith "backpropagation of concatenate not implemented" )
  in
  let initial_grads = GradMap.add GradMap.empty output (ones_like output) in
  let grads =
    List.fold_left
      (fun grads (Var.Any var) -> backprop var grads)
      initial_grads order
  in
  let rec iter_vars : type a b c d.
         (a, b, c, d) input
      -> a Ir.Var.t
      -> b Ir.Var.t
      -> c Hlist.hlist Ir.Var.t
      -> d Hlist.hlist Ir.Var.t =
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
    | Var, E (DiffVar (id, x)) ->
        let grad =
          match GradMap.get grads (DiffVar (id, x)) with
          | Some grad ->
              grad
          | None ->
              failwith "Output does not depend on input"
        in
        E grad :: next
    | Const, E (NoGrad _) ->
        next
    | _ ->
        failwith "should be impossible"
  in
  let outputs = f inputs in
  iter_vars l inputs outputs [outputs]

let grad_and_value f = diff Var f
