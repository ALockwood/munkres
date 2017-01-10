// Copyright 2014 clypd, inc.
//
// see /LICENSE file for more information
// Modified Jan 2017 by Andrew Lockwood to support a mtrix of 64bit floating point values
// Also ditched anything that didn't directly support the goal of returning a minimized cost route for the matrix.

package munkres

import (
	"fmt"
	"math"
)

//FloatMatrix Code
type FloatMatrix struct {
	N int64
	A []float64
}

//NewMatrix will return a pointer to a new FloatMatrix
func NewMatrix(n int64) (m *FloatMatrix) {
	m = new(FloatMatrix)
	m.N = n
	m.A = make([]float64, n*n)
	return m
}

//GetElement will return the element of the matrix at position (i,j)
func (m FloatMatrix) GetElement(i int64, j int64) float64 {
	return m.A[i*m.N+j]
}

//SetElement will set the element of the matrix at position (i,j)
func (m FloatMatrix) SetElement(i int64, j int64, v float64) {
	m.A[i*m.N+j] = v
}

//Print prints all elements of the matrix
func (m *FloatMatrix) Print() {
	var i, j int64
	for i = 0; i < m.N; i++ {
		for j = 0; j < m.N; j++ {
			fmt.Printf("%f ", m.GetElement(i, j))
		}
		fmt.Print("\n")
	}
}

//Munkres Code
const (
	Unset mark = iota
	Starred
	Primed
	zero64 = int64(0)
)

type mark int

type context struct {
	m          *FloatMatrix
	rowCovered []bool
	colCovered []bool
	marked     []mark
	z0row      int64
	z0column   int64
	rowPath    []int64
	colPath    []int64
}

type step interface {
	compute(*context) (step, bool)
}

type step1 struct{}
type step2 struct{}
type step3 struct{}
type step4 struct{}
type step5 struct{}
type step6 struct{}

func newContext(m *FloatMatrix) *context {
	ctx := context{
		m: &FloatMatrix{
			A: make([]float64, m.N*m.N),
			N: m.N,
		},
		rowPath: make([]int64, 2*m.N),
		colPath: make([]int64, 2*m.N),
		marked:  make([]mark, m.N*m.N),
	}
	copy(ctx.m.A, m.A)
	clearCovers(&ctx)
	return &ctx
}

func min(a ...float64) float64 {
	min := math.MaxFloat64
	for _, i := range a {
		if i < min {
			min = i
		}
	}
	return min
}

func (step1) compute(ctx *context) (step, bool) {
	n := ctx.m.N
	for i := zero64; i < n; i++ {
		row := ctx.m.A[i*n : (i+1)*n]
		minval := min(row...)
		for idx := range row {
			row[idx] -= minval
		}
	}
	return step2{}, false
}

func clearCovers(ctx *context) {
	n := ctx.m.N
	ctx.rowCovered = make([]bool, n)
	ctx.colCovered = make([]bool, n)
}

func (step2) compute(ctx *context) (step, bool) {
	n := ctx.m.N
	for i := zero64; i < n; i++ {
		rowStart := i * n
		for j := zero64; j < n; j++ {
			pos := rowStart + j
			if (ctx.m.A[pos] == 0) &&
				!ctx.colCovered[j] && !ctx.rowCovered[i] {
				ctx.marked[pos] = Starred
				ctx.colCovered[j] = true
				ctx.rowCovered[i] = true
			}
		}
	}
	clearCovers(ctx)
	return step3{}, false
}

func (step3) compute(ctx *context) (step, bool) {
	n := ctx.m.N
	count := zero64
	for i := zero64; i < n; i++ {
		rowStart := i * n
		for j := zero64; j < n; j++ {
			pos := rowStart + j
			if ctx.marked[pos] == Starred {
				ctx.colCovered[j] = true
				count++
			}
		}
	}
	if count >= n {
		return nil, true
	}

	return step4{}, false
}

func findAZero(ctx *context) (int64, int64) {
	row := int64(-1)
	col := int64(-1)
	n := ctx.m.N
Loop:
	for i := zero64; i < n; i++ {
		rowStart := i * n
		for j := zero64; j < n; j++ {
			if (ctx.m.A[rowStart+j] == 0) &&
				!ctx.rowCovered[i] && !ctx.colCovered[j] {
				row = i
				col = j
				break Loop
			}
		}
	}
	return row, col
}

func findStarInRow(ctx *context, row int64) int64 {
	n := ctx.m.N
	for j := zero64; j < n; j++ {
		if ctx.marked[row*n+j] == Starred {
			return j
		}
	}
	return -1
}

func (step4) compute(ctx *context) (step, bool) {
	starCol := int64(-1)
	for {
		row, col := findAZero(ctx)
		if row < 0 {
			return step6{}, false
		}
		n := ctx.m.N
		pos := row*n + col
		ctx.marked[pos] = Primed
		starCol = findStarInRow(ctx, row)
		if starCol >= 0 {
			col = starCol
			ctx.rowCovered[row] = true
			ctx.colCovered[col] = false
		} else {
			ctx.z0row = row
			ctx.z0column = col
			break
		}
	}
	return step5{}, false
}

func findStarInCol(ctx *context, col int64) int64 {
	n := ctx.m.N
	for i := zero64; i < n; i++ {
		if ctx.marked[i*n+col] == Starred {
			return i
		}
	}
	return -1
}

func findPrimeInRow(ctx *context, row int64) int64 {
	n := ctx.m.N
	for j := zero64; j < n; j++ {
		if ctx.marked[row*n+j] == Primed {
			return j
		}
	}
	return -1
}

func convertPath(ctx *context, count int) {
	n := ctx.m.N
	for i := 0; i < count+1; i++ {
		r, c := ctx.rowPath[i], ctx.colPath[i]
		offset := r*n + c
		if ctx.marked[offset] == Starred {
			ctx.marked[offset] = Unset
		} else {
			ctx.marked[offset] = Starred
		}
	}
}

func erasePrimes(ctx *context) {
	n := ctx.m.N
	for i := zero64; i < n; i++ {
		rowStart := i * n
		for j := zero64; j < n; j++ {
			if ctx.marked[rowStart+j] == Primed {
				ctx.marked[rowStart+j] = Unset
			}
		}
	}
}

func (step5) compute(ctx *context) (step, bool) {
	count := 0
	ctx.rowPath[count] = ctx.z0row
	ctx.colPath[count] = ctx.z0column
	var done bool
	for !done {
		row := findStarInCol(ctx, ctx.colPath[count])
		if row >= 0 {
			count++
			ctx.rowPath[count] = row
			ctx.colPath[count] = ctx.colPath[count-1]
		} else {
			done = true
		}

		if !done {
			col := findPrimeInRow(ctx, ctx.rowPath[count])
			count++
			ctx.rowPath[count] = ctx.rowPath[count-1]
			ctx.colPath[count] = col
		}
	}
	convertPath(ctx, count)
	clearCovers(ctx)
	erasePrimes(ctx)
	return step3{}, false
}

func findSmallest(ctx *context) float64 {
	n := ctx.m.N
	minval := math.MaxFloat64
	for i := zero64; i < n; i++ {
		rowStart := i * n
		for j := zero64; j < n; j++ {
			if (!ctx.rowCovered[i]) && (!ctx.colCovered[j]) {
				a := ctx.m.A[rowStart+j]
				if minval > a {
					minval = a
				}
			}
		}
	}
	return minval
}

func (step6) compute(ctx *context) (step, bool) {
	n := ctx.m.N
	minval := findSmallest(ctx)
	for i := zero64; i < n; i++ {
		rowStart := i * n
		for j := zero64; j < n; j++ {
			if ctx.rowCovered[i] {
				ctx.m.A[rowStart+j] += minval
			}
			if !ctx.colCovered[j] {
				ctx.m.A[rowStart+j] -= minval
			}
		}
	}
	return step4{}, false
}

//GetMunkresMinScore returns the sum of the elements that comprise the lowest cost path
func GetMunkresMinScore(m *FloatMatrix) float64 {
	ctx := newContext(m)

	var stp step
	stp = step1{}
	for {
		nextStep, done := stp.compute(ctx)

		if done {
			break
		}
		stp = nextStep
	}

	var sumMinCost float64
	for markedIdx, markedVal := range ctx.marked {
		sumMinCost += float64(markedVal) * m.A[markedIdx]
	}

	return sumMinCost
}
