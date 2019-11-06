max(X,Y,X):- X >= Y,!.
max(X,Y,Y). 

member_of(X,[X|_]).
member_of(X,[H|T]):-member(X,T).

concat([],L,L).
concat([H1|L1],L2,[H1|L3]):-concat(L1,L2,L3). 

alternate_sum([X],X):-!.
alternate_sum([A,B|L],S):-alternate_sum(L,S1), S is S1+A-B. 

elim_one([H|T],H,T):-!.
elim_one([H|T],X,[H|elim_one(T,X)]).

elim_all([],_,[]):-!.
elim_all([H|T],H,L):-elim_all(T,H,L),!.
elim_all([H|T],X,[H|L]):-elim_all(T,X,L).

reverse_list([X],[X]):-!.
reverse_list([H|T],L1):-reverse_list(T,L),append(L,[H],L1).

occurences([],_,0):-!.
occurences([X|T],X,Counter1):-occurences(T,X,Counter),Counter1 is Counter+1,!.
occurences([_|T],X,Counter):-occurences(T,X,Counter).

insert_elem(L,X,0,[X|L]):-!.
insert_elem([H|T],X,NR,[H|T1]):-NR1 is NR-1, insert_elem(T,X,NR1,T1).

merge([],L,L):-!.
merge(L,[],L):-!.
merge([H1|T1],[H2|T2],[H1|L]):-H1=<H2,merge(T1,[H2|T2],L),!.
merge(L,[H2|T2],[H2|L1]):-merge(L,T2,L1),!.  


