////// Heap structure
//////
////// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, ..
//////  |  |  |  |        |  |                         |  |  
//////  +  +--+  +--------+  +--------------------------  +-----------------------------------------
//////  
//////
//////
//////                                         0
//////                                        / \
//////                                       /   \
//////                                      /     \
//////                                     /       \
//////                                    /         \
//////                                   /           \
//////                                  /             \
//////                                 /               \
//////                                /                 \
//////                               1                   2
//////                              / \                 / \
//////                             /   \               /   \
//////                            /     \             /     \
//////                           /       \           /       \
//////                          3         4         5         6
//////                         / \       / \       / \       / \
//////                        /   \     /   \     /   \     /   \ 
//////                       /     \   /     \   /     \   /     \ 
//////                      7      8  9     10  11    12  13     14
//////
//////
////
////fn right<T: std::cmp::Ord>(key: T, heap: Vec<T>) {
////
////}
////
////fn left<T: std::cmp::Ord>(key: T, heap: Vec<T>) {
////
////}
////
////fn parent<T: std::cmp::Ord>(key: T, heap: Vec<T>) {
////
////}
////
////fn insert<T: std::cmp::Ord>(key: T, heap: &mut Vec<T>) {
////    // last node = key, r = lastnode
////    // while (r != root) do {
////    //      if ( r< Parent(r) ) then {
////    //          swap(r, Parent(r))
////    //      }
////    //      else break
////    //      }
////    // }
////}
////
////fn delete<T: std::cmp::Ord>(key: T, heap: &mut Vec<T>) {
//////    key = lastnode, lastnode = null, p = root
//////    while (Left(p) != null or Right(p) != null) do {
//////        c = min(Left(p), Right(p))
//////        if ( p < c) { break }
//////        else { swap(p, c) }
//////    }
////}
