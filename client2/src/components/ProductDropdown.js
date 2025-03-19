import Select from 'react-select';
const ProductDropdown = ({ products, selectedProduct, setSelectedProduct }) => {
    const options = products.map((product) => ({
      value: product.product_id,
      label: (
        <div style={{ display: "flex", alignItems: "center" }}>
          <img src={product.image_url} alt={product.name} style={{ width: 50, height: 30, marginRight: 10 }} />
          {product.name} (${product.base_price})
        </div>
      ),
    }));
    return (
      <Select
        options={options}
        value={options.find((option) => option.value === selectedProduct)}
        onChange={(selected) => setSelectedProduct(selected.value)}
      />
    );
  };

export default ProductDropdown;
