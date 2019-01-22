

import Jama.Matrix;

public class Eigen {
	
	public double[][] array = null;
	public double[][] eigenValue = null;
	public double[][] eigenVector = null;
	
	public Eigen(double[][] tempArray,int size) {
		array = tempArray;
		eigenValue = new double[size][size];
		eigenVector = new double[size][size];
	}
	
	public void getEigenValueAndVector() {
		//定义一个矩阵
		Matrix A = new Matrix(array);
		//由特征值组成的对角矩阵
		eigenValue=A.eig().getD().getArray();
		//每一列对应的是一个特征向量
		eigenVector=A.eig().getV().getArray();
	}
	
	public static void main(String args[]) {
		double [][] array = {
				{-1,1,0},
				{-4,3,0},
				{1 ,0,2}};
		Eigen e = new Eigen(array, 3);
		e.getEigenValueAndVector();
		for(int i=0;i<3;i++) {
			for(int j=0;j<3;j++) {
				System.out.print(e.eigenValue[i][j]+" ");
			}
			System.out.println();
		}
		for(int i=0;i<3;i++) {
			for(int j=0;j<3;j++) {
				System.out.print(e.eigenVector[i][j]+" ");
			}
			System.out.println();
		}
	}

}
