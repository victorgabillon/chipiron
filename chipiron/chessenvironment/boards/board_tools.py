
def convertLine(line,index):
    if len(line) ==0:
        return ''

    count = 0
    while index+count <8 and line[count] == '1':
       count = count + 1

    if count == 0:
      return line[0] + convertLine(line[1:],index+1)
    else:
      return str(count) + convertLine(line[count:],index+count)

def convertToFen(asciiBoard):

    list = asciiBoard.splitlines()
    fen = ''
    list2 = list[:-1]
    for line in list2:

        fen  = fen +  convertLine(line,0) + '/'
    fen = fen[:-1]
    fen = fen+' '+list[-1]
    return fen
