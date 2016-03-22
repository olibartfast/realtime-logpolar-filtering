#include "Viewer.h"
#include "Benchmark.h"

int main()
{
system(CLRSCR);
cout<<"1. Elabora in real time"<<endl;
cout<<"2. Benchmark del programma"<<endl;
Image img;
int x;
cin>>x;
switch(x)
{
  case 1:
	{
	Viewer view;
	view.SetImage(&img);
	view.show();
	}
	break;
  case 2:
	{
	Benchmark benchmark(&img);
	benchmark.ReadImg();
	benchmark.SaveImg();
	benchmark.Run();
	}
	break;
}

return 0;
}



