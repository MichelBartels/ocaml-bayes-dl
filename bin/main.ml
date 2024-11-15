open Iree_bindings
open Dsl

(* let f' = Backpropagate.diff [Var; Var] (fun [x; y] -> (x * x) + y) *)

(* let g x = *)
(*   let [[grad1; grad2]; value] = f' x in *)
(*   grad1 + grad2 + value *)

(* let g = fn (List_type [Tensor_type ([], F32); Tensor_type ([], F32)]) g *)

let g' =
  Backpropagate.diff [Var; Var; Var] (fun [x; y; z] -> matmul (matmul x y) z)

let g' =
  fn
    (List_type
       [ Tensor_type ([2; 1; 2], F32)
       ; Tensor_type ([2; 2], F32)
       ; Tensor_type ([2; 1], F32) ] )
    g'

let g' = Ir.compile g'

let () =
  print_endline g' ;
  Compile.compile g' "out.vmfb"
