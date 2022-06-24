import math

def typos(DIM_1, DIM_2):
    total = 0
    # input
    # total += DIM_1 * DIM_2 * 1  #dimensions of spectogram * channel
    # First Conv2d
    dim1 = math.floor(((DIM_1 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((DIM_2 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Second Conv2
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Third Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Fourth Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1

    return dim1 * dim2 * 64  # returns Flatten output

def typos2(DIM_1, DIM_2):
    total = 0
    # input
    # total += DIM_1 * DIM_2 * 1  #dimensions of spectogram * channel
    # First Conv2d
    dim1 = math.floor(((DIM_1 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((DIM_2 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Second Conv2
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Third Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Fourth Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1

    return dim1  # returns Flatten output

def typos3(DIM_1, DIM_2):
    total = 0
    # input
    # total += DIM_1 * DIM_2 * 1  #dimensions of spectogram * channel
    # First Conv2d
    dim1 = math.floor(((DIM_1 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((DIM_2 + 2*1 - 3)/1)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Second Conv2
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Third Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 2)) + 1  # (input_dimension + 2* padding)/stride + 1
    # Fourth Conv2d
    dim1 = math.floor(((dim1 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1
    dim2 = math.floor(((dim2 + 2 * 1 - 3) / 1)) + 1  # (input_dimension + 2* padding)/stride + 1

    return dim2

print(typos(DIM_1=256,DIM_2=173))
print(typos2(256,173))
print(typos3(256,173))