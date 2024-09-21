def fen_to_vector(fen):
    fen_parts = fen.split(" ")
    if fen_parts[1] == "b":
        
        fen_parts[0] = fen_parts[0][::-1].swapcase()
        
        fen_parts[2] = fen_parts[2].swapcase()
        #fen_parts[1] = 1
    # else:
    #     #fen_parts[1] = 0
    position="0," #special token
    piece_dict = {" ":"1,", "p":"2,", "n":"3,", "b":"4,", "r":"5,", "q":"6,", "k":"7,", "P":"8,", "N":"9,", "B":"10,", "R":"11,", "Q":"12,", "K":"13,"}
    for row in fen_parts[0].split("/"):
        for square in row:
            if square.isalpha():
                position+=piece_dict[square]
            else:
                position+=int(square)*"1,"
    castling_rights = fen_parts[2]
    position += "1," if "K" in castling_rights else "0,"
    position += "1," if "Q" in castling_rights else "0,"
    position += "1," if "k" in castling_rights else "0,"
    position += "1," if "q" in castling_rights else "0,"
    en_passant = fen_parts[3]
    if en_passant == "-":
        position += "0," * 9
    else:
        position += "1,"
        file_index = ord(en_passant[0]) - 97
        position += "0," * file_index
        position += "1,"
        position += "0," * (7 - file_index)


    return position[:-1]

print(fen_to_vector("2kr1bnr/p1p3pp/2nppq2/4p3/Pp1PP3/2N2N1P/1PPB1PP1/R2Q1RK1 b - - 0 1"))
print(fen_to_vector("2kr1bnr/p1p3pp/2nppq2/Pp2p3/3PP3/2N2N1P/1PPB1PP1/R2Q1RK1 w Qk b6 0 1"))