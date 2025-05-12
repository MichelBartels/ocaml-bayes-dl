open Dsl

(* (input_type, old_output_type, list end, output list) *)
type (_, _, _, _) input =
  | Var : ('a, 'b, 'c, 'a * 'c) input
  | Const : ('a, 'b, 'c, 'c) input
  | [] : (unit Hlist.hlist, 'a, 'b, unit Hlist.hlist * 'b) input
  | ( :: ) :
      ('a, 'b, 'c, 'd) input
      * ('e Hlist.hlist, 'b, 'f, 'c Hlist.hlist * 'f) input
      -> (('a * 'e) Hlist.hlist, 'b, 'f, 'd Hlist.hlist * 'f) input

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
      | DotProduct (x, y, _, _, _, _)
      | Select (_, x, y) ->
          let visited, order = loop visited order x in
          let visited, order = loop visited order y in
          let var = Var.Any var in
          (visited, var :: order)
      | Negate x
      | Abs x
      | Ln x
      | Ln_1_plus x
      | Exponential x
      | BroadcastInDim (x, _)
      | Transpose (x, _)
      | Reshape (x, _)
      | Sin x
      | Cos x
      | Tanh x
      | Sqrt x
      | OptimizationBarrier x ->
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
    (a, b) Var.u -> (c, d) Var.u -> (c, d) Var.u =
 fun x y ->
  match (Var.value_type x, Var.value_type y) with
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

let rec assert_same_type : type a b. a Var.t -> b Var.t -> b Var.t =
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
            Some (Any (grad +$ grad')) )
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
       (a, (b, d) Hlist.element, (b, d) Hlist.element * unit, c) input
    -> (a Var.t -> (b, d) Hlist.element Var.t)
    -> a Var.t
    -> c Hlist.hlist Var.t =
 fun l f inputs ->
  let rec wrap_inputs : type a b c d. (a, b, c, d) input -> a Var.t -> a Var.t =
   fun l1 l2 ->
    match (l1, l2) with
    | [], [] ->
        []
    | x :: xs, y :: ys ->
        let input = wrap_inputs x y in
        let inputs = wrap_inputs xs ys in
        input :: inputs
    | Var, x ->
        Var.map
          { f=
              (fun x ->
                let id = Var.new_id () in
                DiffVar (id, x) ) }
          x
    | Const, x ->
        Var.map {f= (fun x -> NoGrad x)} x
  in
  let inputs = wrap_inputs l inputs in
  let (E output) = f inputs in
  let order = topological_order output in
  let backprop : type a b. (a, b) Var.u -> GradMap.t -> GradMap.t =
   fun var grads ->
    match GradMap.get grads var with
    | None ->
        grads
    | Some grad -> (
      match var with
      | Var.Add (v1, v2) ->
          let grads = GradMap.add grads v1 grad in
          GradMap.add grads v2 grad
      | Subtract (v1, v2) ->
          let grads = GradMap.add grads v1 grad in
          GradMap.add grads v2 ~-$grad
      | Multiply (v1, v2) ->
          let grads = GradMap.add grads v1 (grad *$ v2) in
          GradMap.add grads v2 (grad *$ v1)
      | Negate v ->
          GradMap.add grads v ~-$grad
      | Abs _ ->
          failwith "abs not yet implemented"
      | Ln v ->
          GradMap.add grads v (grad /$ v)
      | Ln_1_plus v ->
          GradMap.add grads v (grad /$ (v +$ ones_like v))
      | Exponential v ->
          GradMap.add grads v (grad *$ exp v)
      | Pow (v1, v2) ->
          let grads = GradMap.add grads v1 (grad *$ var *$ v2 /$ v1) in
          GradMap.add grads v2 (grad *$ ln v1 *$ var)
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
            let var_shape = Var.shape var in
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
            let grad_shape = Var.shape grad in
            let const_shape = Var.shape const in
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
              Var.DotProduct
                ( const
                , grad
                , const_contracting_dims
                , grad_contracting_dims
                , const_batching_dims
                , grad_batching_dims )
            in
            Var.Transpose (prod, permutation)
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
          let grads = GradMap.add grads v1 (grad /$ v2) in
          GradMap.add grads v2 (grad *$ ~-$v1 /$ (v2 *$ v2))
      | BroadcastInDim (var, dims) -> (
        match Var.value_type var with
        | _, Tensor.F32 ->
            let reduced_grad =
              sum ~axes:(List.init (List.length dims) Fun.id) grad
            in
            GradMap.add grads var reduced_grad
        | _ ->
            failwith "can only differentiate broadcasting of f32 tensors" )
      | Transpose (var, permutation) ->
          let inverse_permutation =
            List.init (List.length permutation) (fun i ->
                List.find_index (fun x -> x = i) permutation |> Option.get )
          in
          let grad = Var.Transpose (grad, inverse_permutation) in
          GradMap.add grads var grad
      | Tanh var ->
          GradMap.add grads var
            (grad *$ (Dsl.ones_like var -$ (tanh var *$ tanh var)))
          (* Jax uses different backpropagation that is more numerically precise, but also slower *)
      | Sum (var, dimensions) ->
          let var_shape = Var.shape var in
          let new_dims =
            List.filteri (fun i _ -> List.mem i dimensions) var_shape
          in
          let new_grad = Var.BroadcastInDim (grad, new_dims) in
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
          let grad = Var.Transpose (new_grad, perm) in
          GradMap.add grads var grad
      | Reshape (var, _) ->
          let grad = reshape (Var.shape var) grad in
          GradMap.add grads var grad
      | Sin var ->
          GradMap.add grads var (grad *$ cos var)
      | Cos var ->
          GradMap.add grads var (grad *$ ~-$(sin var))
      | Concatenate _ ->
          failwith "backpropagation of concatenate not implemented"
      | Select (cond, x, y) ->
          let zero_grads = zeros_like grad in
          let x_grad = select cond grad zero_grads in
          let y_grad = select cond zero_grads grad in
          let grads = GradMap.add grads x x_grad in
          GradMap.add grads y y_grad
      | Sqrt var ->
          GradMap.add grads var
            (grad /$ ((ones_like var +$ ones_like var) *$ sqrt var))
      | OptimizationBarrier var ->
          GradMap.add grads var grad )
  in
  let initial_grads = GradMap.add GradMap.empty output (ones_like output) in
  let grads =
    List.fold_left
      (fun grads (Var.Any var) -> backprop var grads)
      initial_grads order
  in
  let rec iter_vars : type a b c d.
         (a, b, c, d) input
      -> a Var.t
      -> b Var.t
      -> c Hlist.hlist Var.t
      -> d Hlist.hlist Var.t =
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
  iter_vars l inputs (E output) [E output]

let grad_and_value f = diff Var f

let%expect_test "backprop_add" =
  let x = full F32 1.0 [2; 2] in
  let y = full F32 2.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (x +$ y)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| const(1.000000e+00) |}]

let%expect_test "backprop_subtract" =
  let x = full F32 1.0 [2; 2] in
  let y = full F32 2.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (x -$ y)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| const(1.000000e+00) |}]

let%expect_test "backprop_multiply" =
  let x = full F32 2.0 [2; 2] in
  let y = full F32 3.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (x *$ y)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| (const(1.000000e+00) * const(3.000000e+00)) |}]

let%expect_test "backprop_divide" =
  let x = full F32 6.0 [2; 2] in
  let y = full F32 2.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (x /$ y)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| (const(1.000000e+00) / const(2.000000e+00)) |}]

let%expect_test "backprop_negate" =
  let x = full F32 1.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E ~-$x) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| (-const(1.000000e+00)) |}]

let%expect_test "backprop_exp" =
  let x = full F32 1.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (exp x)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| (const(1.000000e+00) * exp(diff5(const(1.000000e+00)))) |}]

let%expect_test "backprop_ln" =
  let x = full F32 2.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (ln x)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| (const(1.000000e+00) / diff6(const(2.000000e+00))) |}]

let%expect_test "backprop_sin" =
  let x = full F32 0.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (sin x)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| (const(1.000000e+00) * cos(diff7(const(0.000000e+00)))) |}]

let%expect_test "backprop_cos" =
  let x = full F32 0.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (cos x)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| (const(1.000000e+00) * (-sin(diff8(const(0.000000e+00))))) |}]

let%expect_test "backprop_tanh" =
  let x = full F32 0.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (tanh x)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect
    {| (const(1.000000e+00) * (const(1.000000e+00) - (tanh(diff9(const(0.000000e+00))) * tanh(diff9(const(0.000000e+00)))))) |}]

let%expect_test "backprop_sqrt" =
  let x = full F32 4.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (sqrt x)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect
    {| (const(1.000000e+00) / ((const(1.000000e+00) + const(1.000000e+00)) * sqrt(diff10(const(4.000000e+00))))) |}]

let%expect_test "backprop_sum" =
  let x = full F32 1.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (sum ~axes:[0] x)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| transpose(broadcast(const(1.000000e+00), 2), 0,1) |}]

let%expect_test "backprop_transpose" =
  let x = full F32 1.0 [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (transpose x [1; 0])) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| transpose(const(1.000000e+00), 1,0) |}]

let%expect_test "backprop_broadcast" =
  let x = full F32 1.0 [2] in
  let [E grad; _] =
    diff Var (fun (E x) -> E (Var.BroadcastInDim (x, [2; 2]))) (E x)
  in
  print_endline (Var.to_string grad) ;
  [%expect {| sum(const(1.000000e+00), 0,1) |}]

let%expect_test "backprop_select" =
  let x = full F32 1.0 [2; 2] in
  let y = full F32 2.0 [2; 2] in
  let cond = full I1 true [2; 2] in
  let [E grad; _] = diff Var (fun (E x) -> E (select cond x y)) (E x) in
  print_endline (Var.to_string grad) ;
  [%expect {| select(const(true), const(1.000000e+00), const(0.000000e+00)) |}]
