open Iree_bindings
open Dsl

let sigmoid x = Dsl.(ones_like x / (ones_like x + exp (zeros_like x - x)))

let g' = Backpropagate.diff [Var] (fun [x] -> sigmoid x)

let g' =
  fn
    (List_type [Tensor_type ([2; 2], F32)])
    g'

let g' = Ir.compile g'

let () =
  print_endline g' ;
  Compile.compile g' "out.vmfb"
