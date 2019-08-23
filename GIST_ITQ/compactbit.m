function B = compactbit(U)
    bits = U>0;
    U(bits) = 1;
    fbits = U<=0;
    U(fbits) = 0;
    B = U;
