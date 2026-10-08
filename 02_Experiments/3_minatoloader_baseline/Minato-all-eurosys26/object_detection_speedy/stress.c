#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_SIZE 1 << 12

int main()
{
	int nb_pg = 10*1024*1024, step;
	int *m = malloc(nb_pg*PAGE_SIZE), *p;
        //while(1)
        //{
	for (int i = 0; i < nb_pg; i++)
	{
		step = i*PAGE_SIZE;
		p = (int *)malloc(step);
		p[step+1] = 12;
		//memset(m+step,0,PAGE_SIZE);
	}
        //memset(m,0,size);
        //}
        return 0;
}
