
prop_clause(p /\ (not(q) \/ r), S).
S = [[p],[not q,r]];
no
prop_clause(p,R).
R = [[p]];
no
*/
  
/* 2 (a). Resolution */

/* Idea: 
   Recursively choose two different clauses from the input list,
   Resolve them and add the resolvent to input list.
   If we get a [] empty list in any step, report true.
   If we have exhausted all possible combinations of input clauses,
   report no.
*/

member(Element,[Element|_List]).
member(Element,[_First|Tail]) :- member(Element,Tail).

delete(X,[Fi|Li],Lo) :-
  Fi = X,
  Li = Lo.
delete(X,[Fi|Li],[Fo|Lo]) :-
  Fi = Fo,
  Fi \= X,
  delete(X,Li,Lo).

/* Sort individual clauses */

sortClauses([], []).
sortClauses([H|List], [H1|Out]) :-
  sort(H,H1),
  sortClauses(List, Out).                                          

/* Sort the independent clauses of input list and call resolve */
/* Why do we sort individual clauses ?
       Since we check whether a clause is already a member of input list, 
   we need to keep the clauses uniform everywhere. Hence the sorting. E.g
   [p,q] is same as that of [q,p]. */

resolve(List) :-
  sortClauses(List, L),              
  do_resolve(L).

/* Find a resolvent and if it is empty, the given input is unsatisfiable.
   Else append the resolvent to the input list and continue resolving */

do_resolve(List) :-                       
  once(get_resolvent(List, Resolvent)),
  (Resolvent = []
  -> true
  ;resolve([Resolvent|List])
  ).

/* Taking two clauses A and B from List, and resolve them */

get_resolvent(List, Resolvent) :-
  member(A, List),
  member(B, List),
  A \= B,
  res(A, B, T),
  (T = false
  -> false
  ;(member(T, List)
  -> false
  ; Resolvent = T)
  ).

/* We have two clauses A and B here. We try to find an atomic element in both
   the clauses which is of the form  x, not(x) and cancel them. Append the rest
   of the atoms to form a resolvent. Then sort the final resolvent */

res(A, B, T) :-
  member(A1, A),
  (A1 = X
  -> B1 = not(X)
  ; not(B1) = A1),
  (member(B1, B)
  -> delete(A1, A, TempA),
     delete(B1, B, TempB),
     append(TempA, TempB, Ans),
     sort(Ans, T)
  ; false
  ).       

/* Test run
resolve([[p],[not p]]).
yes
resolve([[p,not(q)],[q, not(p)]]).
no
resolve([[p, r], [q, not(r)], [not(p)], [not(q)]]).
yes
*/

/* 2(b) printing resolvents and source clauses */
/* Same logic as 2 (a) */

resolve_pr(List) :-
  sortClauses(List, L),              
  do_resolve_pr(L).

do_resolve_pr(List) :-                       
  once(get_resolvent_pr(List, Resolvent)),
  (Resolvent = []
  -> true
  ; resolve_pr([Resolvent|List])
  ).

/* Printing once we resolve something and get a valid resolvent */

get_resolvent_pr(List, Resolvent) :-
  member(A, List),
  member(B, List),
  A \= B,
  res(A, B, T),
  (T = false
  -> false
  ;(member(T, List)
  -> false
  ;
  Resolvent = T),
  write('Source 1 : '),writeln(A),
  write('Source 2 : '),writeln(B), 
  write('Resolvent : '),writeln(Resolvent),writeln('')
  ).

/* Test run
resolve_pr([[p, r], [q, not(r)], [not(p)], [not(q)]]).
Source 1 : [p,r]
Source 2 : [q,not r]
Resolvent : [p,q]
Source 1 : [p,q]
Source 2 : [not p]
Resolvent : [q]
Source 1 : [q]
Source 2 : [not q]
Resolvent : []
*/

/* 2c. An interesting Example */

/* Lets give some meaning to our propositional symbols.
     s = Surya is sleeping
     hw = Surya is doing logic hw
     f = Surya is done with logic hw
  Lets consider the following clauses
   1. [Surya is sleeping, Surya is doing logic hw]
      [s,hw]
   2. [Surya is not doing logic hw, Surya is done with logic hw] 
      [not(hw), f]
   3. [Surya is not done with logic hw]
      [not(f)]
   4. [Surya is not sleeping]
      [not(s)]
      Test Run:
      resolve_pr([[s,hw],[not(hw),f],[not(f)],[not(s)]]).
      Source 1 : [hw,s]
      Source 2 : [f,not hw]
      Resolvent : [f,s]
      Source 1 : [f,s]
      Source 2 : [not f]
      Resolvent : [s]
      Source 1 : [s]
      Source 2 : [not s]
      Resolvent : []
      Lets Analyse,
           First clause says, either Surya is sleeping or He is doing logic hw.
           Second one says, either Surya is not doing logic hw or he is done with it.
           So when we resolve these two, We get a clause which says either Surya is either done with hw or he is sleeping [f, s].
           
           We resolve [f,s] with "Surya is not done with logic hw" and we get "Surya is sleeping".
           Resolving "Surya is sleeping" with the clause which says "Surya is not sleeping" give us an empty list [].
In short:
   From the 1st and 2nd clauses, either I will be happily sleeping or I burn the midnight oil to get done with the homework. 
   But in the 3rd and 4th clauses the claim that "neither I'm sleeping nor I'm done with the homework" makes the whole input unsatisfiable. :) 
*/

do_resolve(List):-
    (once(calculate_resolvent(List,Resolvent)),
    Resolvent = false -> 
        writeOutput(Resolvent);
    (Resolvent = [] -> 
        writeOutput(Resolvent);
        sortByLength([Resolvent|List],L),
        do_resolve(L))
    ).