#include <iostream>
#include <omp.h>

int main()
{
	int numberOfThreads = 6;  

#pragma omp parallel num_threads(numberOfThreads)  
	{
		int num = omp_get_thread_num(); //получаем номер потока параллельно
#pragma omp critical
		{
			std::cout << "Message from thread " << num << std::endl; //доступ к консоли последовательный
		}
	}
}