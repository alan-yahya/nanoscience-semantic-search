const Dropdown = ({ label, options, selectedValue, onChange }) => {
    return (
        <div className="settings-group">
            <label>{label}</label>
            <select className="form-select form-select-sm" value={selectedValue} onChange={onChange}>
                {options.map((option) => (
                    <option key={option.value} value={option.value}>
                        {option.label}
                    </option>
                ))}
            </select>
        </div>
    );
};

export default Dropdown;
