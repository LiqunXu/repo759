
#include <iostream>
#include <cstdio>
using namespace std;

int main(int argc, char *argv[]){
	int x = atoi(argv[1]);
	for (int i = 0; i <= x; i++) {
		printf("%d ", i);
	}
	printf("\n");
	for (int i = x; i >= 0; i--){
		cout << i << " ";
	}
	cout << endl;
}
