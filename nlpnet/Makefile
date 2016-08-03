BASEDIR = .

####################
# Directory layout #
####################
SRCDIR = $(BASEDIR)/src
INCLUDEDIR = $(BASEDIR)/include
INCLUDEDIR += -I$(BASEDIR)/eigen
INCLUDEDIR += -I$(BASEDIR)/json-cpp/include
LIBDIR = $(BASEDIR)/lib
DISTDIR = $(BASEDIR)/dist
TESTDIR = $(BASEDIR)/test

TEST_SOURCES = $(wildcard $(TESTDIR)/*.cpp)
TEST_OBJS = $(patsubst %.cpp,%.o,$(TEST_SOURCES))
TEST_EXES = $(patsubst %.cpp,%,$(TEST_SOURCES))

.PHONY: all clean 

####################
# Compiler options #
#    F_ -- FLAG    #
####################
CC = g++

# Compiler search options
F_INCLUDE = -I$(INCLUDEDIR)
#F_LIB = -L $(LIBDIR) -lANN -lcrypto -lpthread -lm -lz
F_LIB = -L $(BASEDIR)/json-cpp/lib -ljson_libmt -lcrypto -static -lpthread -lm -lz  

CFLAGS = -O3 -std=c++11 
CPPFLAGS = $(CFLAGS) $(F_INCLUDE)

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst %.cpp,%.o,$(SOURCES))

tool: all $(TEST_EXES)

$(TEST_EXES): %: %.o $(OBJS)
	$(CC) -o $@ $< $(OBJS) $(F_LIB)
    
$(OBJS): %.o: %.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

$(TEST_OBJS): %.o: %.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS) 

clean:
	rm -f $(OBJS) $(TEST_OBJS) $(TEST_EXES)
