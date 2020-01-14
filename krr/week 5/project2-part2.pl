use_module(library(lists)).

program:-
    readKB(KB,Goal),
    performReasoning(KB,Goal).

readKB(L,Goal) :-
    see('C:\\master\\krr\\week 5\\kb2.txt'),
    read(L),
    read(G),
    (G = 'patient has pneumonia' -> Goal = [p],! ; Goal = [n(p)],!),
    read(end_of_file),
    seen.

performReasoning(KB, Goal):-
    displayMenu(WM),
    do_forwardchain(KB,WM,Goal),
    write('Stop?'),nl,
    read(Stop),
    (Stop \= 'stop' -> performReasoning(KB, Goal) ; !). 

displayMenu(WM):-
    write('What is patient temperature?'),nl,
    read(Temp),
    write('For how many days has the patient been sick?'),nl,
    read(Sick),
    write('Has patient muscle pain?'),nl,
    read(Muscle),
    write('Has patient cough?'),nl,
    read(Cough),
    buildWM(WM,Temp,Sick,Muscle,Cough).

buildWM(WM,Temp, Sick,Muscle,Cough):-
    (Temp > 38 -> append([], [t], KB1) ; append([],[n(t)],KB1)),
    (Sick >= 2 -> append(KB1, [s], KB2) ; append(KB1, [n(s)], KB2)),
    (Muscle = 'yes' -> append(KB2, [m], KB3) ; append(KB2, [n(m)], KB3)),
    (Cough = 'yes' -> append(KB3, [c], WM) ; append(KB3, [n(c)], WM)),!.

do_forwardchain(KB,WM,Goal):-
    (forwardchain(KB,WM,Goal)->
        write('YES'),nl;
     write('NO'),nl
    ).


forwardchain(_,WM,Goal):- all_from_first_in_second(Goal,WM),!.
forwardchain(KB,WM,Goal):-
    member([Cond|Effects],KB),
    all_from_first_in_second(Cond, WM),
    eliminateList(Effects,E),
    \+all_from_first_in_second(E,WM),
    append(WM,E,NewWM),
    forwardchain(KB,NewWM,Goal),!.


checkEffect(_,[]):-!.
checkEffect(A,[A|_]):-!,fail.
checkEffect(X,[_|T]):-checkEffect(X,T),!.

all_from_first_in_second(List1, List2) :-
    forall(member(Element,List1), member(Element,List2)).

eliminateList([H|_],H):-!.