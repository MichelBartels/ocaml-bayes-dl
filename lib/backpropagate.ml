(* (input_type, old_output_type, list end, output list) *)
type (_, _, _, _) input =
  | Var : ('a Ir.tensor, 'b, 'c, 'b Ir.Var.t -> 'c) input
  | Const : ('a Ir.tensor, 'b, 'c, 'c) input
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
  let rec backprop : type a. a Ir.Var.t -> int -> a Ir.Var.t =
    Dsl.(
      fun v x ->
        match v with
        | Ir.Var.Add (v1, v2) ->
            backprop v1 x + backprop v2 x
        | Subtract (v1, v2) ->
            backprop v1 x - backprop v2 x
        | Multiply (v1, v2) ->
            (v1 * backprop v2 x) + (v2 * backprop v1 x)
        | Abs v ->
            abs (backprop v x)
        | Argument _ ->
            zeros_like v
        | Compare (a, dir, b) ->
            compare dir (backprop a x) (backprop b x)
        | Constant _ ->
            zeros_like v
        | Random _ ->
            zeros_like v
        | [] ->
            []
        | y :: ys ->
            backprop y x :: backprop ys x
        | DiffVar (id, _) ->
            if Int.equal id x then Dsl.ones_like v else Dsl.zeros_like v
        | DiffConst _ ->
            Dsl.zeros_like v )
  in
  let rec wrap_inputs :
      type a b c d. (a, b, c, d) input -> a Ir.Var.t -> a Ir.Var.t * Ir.id list
      =
   fun l1 l2 ->
    match (l1, l2) with
    | [], [] ->
        ([], [])
    | x :: xs, y :: ys ->
        let input, ids1 = wrap_inputs x y in
        let inputs, ids2 = wrap_inputs xs ys in
        (input :: inputs, ids2 @ ids1)
    | Var, x ->
        let id = Ir.new_id () in
        (DiffVar (id, x), [id])
    | Const, x ->
        (DiffConst x, [])
  in
  let rec iter_vars :
      type a b c d.
         (a, b, c, d) input
      -> b Ir.Var.t
      -> c Ir.VarList.t Ir.Var.t
      -> Ir.id list
      -> d Ir.VarList.t Ir.Var.t * Ir.id list =
   fun l outputs next ids ->
    match (l, ids) with
    | [], ids ->
        ([] :: next, ids)
    | x :: xs, ids ->
        let outputs' :: _, ids = iter_vars xs outputs next ids in
        let output, ids = iter_vars x outputs outputs' ids in
        (output :: next, ids)
    | Var, id :: ids ->
        let output = backprop outputs id in
        (output :: next, ids)
    | Const, ids ->
        (next, ids)
    | _ ->
        print_endline (string_of_int (List.length ids)) ;
        failwith "should be impossible"
  in
  let inputs, ids = wrap_inputs l inputs in
  print_endline (string_of_int (List.length ids)) ;
  let outputs = f inputs in
  fst @@ iter_vars l outputs [outputs] ids
