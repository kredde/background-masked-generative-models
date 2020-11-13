.PHONY: all clean

TAG=oodcoco
DUMMY_TARGET=dummy

all: $(DUMMY_TARGET)

$(DUMMY_TARGET): Dockerfile $(DUMMY_TARGET)
	docker build -t $(TAG) -f $< .
	touch $@

clean:
	rm $(DUMMY_TARGET)
