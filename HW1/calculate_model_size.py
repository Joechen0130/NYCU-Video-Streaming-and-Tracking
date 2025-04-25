from net import my_network

if __name__ == "__main__":
    my_net = my_network()
    params = sum(p.numel() for p in my_net.parameters())
    print(params)