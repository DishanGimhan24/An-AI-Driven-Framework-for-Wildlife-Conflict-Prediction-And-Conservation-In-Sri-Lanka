import React, { useState, useEffect } from 'react';
import {
  View, Text, ScrollView, TouchableOpacity, StyleSheet,
  TextInput, Image, Alert, ActivityIndicator, StatusBar, Switch, Platform,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Picker } from '@react-native-picker/picker';
import DateTimePicker from '@react-native-community/datetimepicker';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import { API_BASE_URL } from '../config/api';

const OFFENCE_TYPES = [
  { label: 'Suspected Poaching', value: 'suspected_poaching' },
  { label: 'Illegal Logging', value: 'illegal_logging' },
  { label: 'Traps or Weapons', value: 'traps_or_weapons' },
  { label: 'Meat or Egg Trade', value: 'meat_trade' },
  { label: 'Other', value: 'other' },
];

export default function ReportScreen({ navigation }) {
  const insets = useSafeAreaInsets();

  const [regions, setRegions] = useState([]);
  const [locations, setLocations] = useState([]);
  const [loadingRegions, setLoadingRegions] = useState(true);
  const [region, setRegion] = useState('');
  const [location, setLocation] = useState('');
  const [offenceType, setOffenceType] = useState('suspected_poaching');
  const [description, setDescription] = useState('');
  const [isAnonymous, setIsAnonymous] = useState(true);
  const [when, setWhen] = useState(null);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [imageUri, setImageUri] = useState(null);
  const [coordinates, setCoordinates] = useState(null);
  const [gettingLocation, setGettingLocation] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [errors, setErrors] = useState({});

  // Fetch regions on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/regions`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const list = data.regions || [];
        setRegions(list);
        if (list.length > 0) setRegion(list[0]);
      } catch {
        Alert.alert('Network Error', 'Could not load regions. Make sure the backend is running.');
      } finally {
        setLoadingRegions(false);
      }
    })();
  }, []);

  // Fetch locations when region changes
  useEffect(() => {
    if (!region) return;
    (async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/locations?region=${encodeURIComponent(region)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const list = data.locations || [];
        setLocations(list);
        if (list.length > 0) setLocation(list[0]);
      } catch {
        setLocations([]);
      }
    })();
  }, [region]);

  async function handlePickImage() {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please allow photo library access to attach evidence.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.8,
    });
    if (!result.canceled && result.assets.length > 0) setImageUri(result.assets[0].uri);
  }

  async function handleTakePhoto() {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please allow camera access to take a photo.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ allowsEditing: true, quality: 0.8 });
    if (!result.canceled && result.assets.length > 0) setImageUri(result.assets[0].uri);
  }

  async function handleGetLocation() {
    setGettingLocation(true);
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission needed', 'Please allow location access to auto-fill coordinates.');
        return;
      }
      const loc = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High });
      setCoordinates({ latitude: loc.coords.latitude, longitude: loc.coords.longitude });
    } catch {
      Alert.alert('Error', 'Could not get GPS location. Please try again.');
    } finally {
      setGettingLocation(false);
    }
  }

  function validate() {
    const errs = {};
    if (!region) errs.region = 'Region is required';
    if (!location) errs.location = 'Location is required';
    if (!description.trim()) errs.description = 'Description is required';
    setErrors(errs);
    return Object.keys(errs).length === 0;
  }

  async function handleSubmit() {
    if (!validate()) return;
    setSubmitting(true);
    try {
      let imageUrl = null;
      if (imageUri) {
        const filename = imageUri.split('/').pop();
        const ext = filename.split('.').pop();
        const formData = new FormData();
        formData.append('file', { uri: imageUri, type: `image/${ext}`, name: filename });
        const uploadRes = await fetch(`${API_BASE_URL}/api/upload`, {
          method: 'POST',
          body: formData,
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        if (!uploadRes.ok) {
          const errBody = await uploadRes.json().catch(() => ({}));
          throw new Error(errBody.detail || 'Image upload failed');
        }
        imageUrl = (await uploadRes.json()).image_url;
      }

      const payload = {
        region, location,
        offence_type: offenceType,
        incident_datetime: when ? when.toISOString() : null,
        description: description.trim(),
        image_url: imageUrl,
        latitude: coordinates?.latitude ?? null,
        longitude: coordinates?.longitude ?? null,
        is_anonymous: isAnonymous,
      };

      const res = await fetch(`${API_BASE_URL}/api/reports`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server error: ${res.status}`);
      }
      const data = await res.json();

      Alert.alert(
        '✓ Report Submitted!',
        `Reference ID: ${data.reference_id}\nStatus: ${data.status}\n\nThank you for helping protect wildlife.`,
        [{ text: 'OK', onPress: () => navigation.navigate('Home') }],
      );
      setDescription(''); setImageUri(null); setCoordinates(null);
      setWhen(null); setOffenceType('suspected_poaching');
      if (regions.length > 0) setRegion(regions[0]);
    } catch (e) {
      Alert.alert('Submission Failed', e.message || 'Something went wrong.', [
        { text: 'Retry', onPress: handleSubmit },
        { text: 'Cancel', style: 'cancel' },
      ]);
    } finally {
      setSubmitting(false);
    }
  }

  function onDateChange(event, selectedDate) {
    if (Platform.OS === 'android') setShowDatePicker(false);
    if (event.type === 'dismissed') return;
    if (selectedDate) {
      const base = when ? new Date(when) : new Date();
      base.setFullYear(selectedDate.getFullYear(), selectedDate.getMonth(), selectedDate.getDate());
      setWhen(base);
    }
  }

  function onTimeChange(event, selectedTime) {
    if (Platform.OS === 'android') setShowTimePicker(false);
    if (event.type === 'dismissed') return;
    if (selectedTime) {
      const base = when ? new Date(when) : new Date();
      base.setHours(selectedTime.getHours(), selectedTime.getMinutes());
      setWhen(base);
    }
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <StatusBar barStyle="light-content" backgroundColor="#0f172a" />
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.pageHeader}>
          <Text style={styles.pageTitle}>📝 Community Report</Text>
          <Text style={styles.pageSubtitle}>Report illegal poaching or wildlife offences. Your report goes to wildlife officers.</Text>
        </View>

        <View style={styles.formCard}>

          {/* Region */}
          <View style={styles.formGroup}>
            <Text style={styles.label}>🌍 Region <Text style={styles.required}>*</Text></Text>
            {loadingRegions ? (
              <ActivityIndicator color="#818cf8" style={{ marginTop: 8 }} />
            ) : (
              <View style={[styles.pickerWrap, errors.region && styles.fieldError]}>
                <Picker
                  selectedValue={region}
                  onValueChange={(v) => { setRegion(v); setErrors(p => ({ ...p, region: undefined })); }}
                  style={styles.picker}
                  dropdownIconColor="#94a3b8"
                  itemStyle={styles.pickerItem}
                >
                  {regions.map(r => <Picker.Item key={r} label={r} value={r} color="#ffffff" />)}
                </Picker>
              </View>
            )}
            {errors.region && <Text style={styles.errorText}>{errors.region}</Text>}
          </View>

          {/* Location */}
          <View style={styles.formGroup}>
            <Text style={styles.label}>📍 General Location <Text style={styles.required}>*</Text></Text>
            <View style={[styles.pickerWrap, errors.location && styles.fieldError]}>
              <Picker
                selectedValue={location}
                onValueChange={(v) => { setLocation(v); setErrors(p => ({ ...p, location: undefined })); }}
                style={styles.picker}
                dropdownIconColor="#94a3b8"
                itemStyle={styles.pickerItem}
              >
                {locations.map(l => <Picker.Item key={l} label={l} value={l} color="#ffffff" />)}
              </Picker>
            </View>
            {errors.location && <Text style={styles.errorText}>{errors.location}</Text>}
          </View>

          {/* GPS */}
          <View style={styles.formGroup}>
            <Text style={styles.label}>📌 Exact GPS Coordinates (Optional)</Text>
            <Text style={styles.hintText}>Tap below to auto-fill your current coordinates.</Text>
            {coordinates ? (
              <View style={styles.coordsRow}>
                <Text style={styles.coordsText}>📍 {coordinates.latitude.toFixed(5)}, {coordinates.longitude.toFixed(5)}</Text>
                <TouchableOpacity onPress={() => setCoordinates(null)}>
                  <Text style={styles.removeLink}>Remove</Text>
                </TouchableOpacity>
              </View>
            ) : (
              <TouchableOpacity style={styles.locationBtn} onPress={handleGetLocation} disabled={gettingLocation} activeOpacity={0.8}>
                {gettingLocation
                  ? <ActivityIndicator size="small" color="#818cf8" />
                  : <Text style={styles.locationBtnText}>📡 Use My Current Location</Text>}
              </TouchableOpacity>
            )}
          </View>

          {/* Offence Type */}
          <View style={styles.formGroup}>
            <Text style={styles.label}>⚠️ Offence Type</Text>
            <View style={styles.pickerWrap}>
              <Picker selectedValue={offenceType} onValueChange={setOffenceType} style={styles.picker} dropdownIconColor="#94a3b8" itemStyle={styles.pickerItem}>
                {OFFENCE_TYPES.map(o => <Picker.Item key={o.value} label={o.label} value={o.value} color="#ffffff" />)}
              </Picker>
            </View>
          </View>

          {/* Date & Time */}
          <View style={styles.formGroup}>
            <Text style={styles.label}>📅 Date & Time (Optional)</Text>
            <View style={styles.dateRow}>
              <TouchableOpacity style={[styles.dateBtn, { flex: 1 }]} onPress={() => setShowDatePicker(true)}>
                <Text style={styles.dateBtnText}>{when ? when.toLocaleDateString() : 'Pick Date'}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.dateBtn, { flex: 1 }]} onPress={() => setShowTimePicker(true)}>
                <Text style={styles.dateBtnText}>{when ? when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'Pick Time'}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.nowBtn} onPress={() => setWhen(new Date())}>
                <Text style={styles.nowBtnText}>Now</Text>
              </TouchableOpacity>
            </View>
            {when && <Text style={styles.datePreview}>✓ {when.toLocaleString()}</Text>}
            {when && <TouchableOpacity onPress={() => setWhen(null)}><Text style={[styles.removeLink, { marginTop: 4 }]}>Clear date</Text></TouchableOpacity>}
          </View>

          {showDatePicker && (
            <DateTimePicker value={when || new Date()} mode="date" display={Platform.OS === 'ios' ? 'spinner' : 'default'} onChange={onDateChange} themeVariant="dark" />
          )}
          {showTimePicker && (
            <DateTimePicker value={when || new Date()} mode="time" display={Platform.OS === 'ios' ? 'spinner' : 'default'} onChange={onTimeChange} themeVariant="dark" />
          )}

          {/* Description */}
          <View style={styles.formGroup}>
            <Text style={styles.label}>📄 Description <Text style={styles.required}>*</Text></Text>
            <TextInput
              style={[styles.textArea, errors.description && styles.fieldError]}
              multiline numberOfLines={5} value={description}
              onChangeText={(t) => { setDescription(t); setErrors(p => ({ ...p, description: undefined })); }}
              placeholder="What happened? Any vehicle number, people count, direction, sounds, or evidence?"
              placeholderTextColor="#475569" textAlignVertical="top"
            />
            {errors.description && <Text style={styles.errorText}>{errors.description}</Text>}
          </View>

          {/* Photo Evidence */}
          <View style={styles.formGroup}>
            <Text style={styles.label}>📸 Photo Evidence (Optional)</Text>
            {imageUri ? (
              <View style={styles.imagePreviewWrap}>
                <Image source={{ uri: imageUri }} style={styles.imagePreview} resizeMode="cover" />
                <TouchableOpacity style={styles.removeImageBtn} onPress={() => setImageUri(null)} hitSlop={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                  <Text style={styles.removeImageBtnText}>✕</Text>
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.photoButtonsRow}>
                <TouchableOpacity style={styles.photoBtn} onPress={handlePickImage} activeOpacity={0.8}>
                  <Text style={styles.photoBtnText}>📁 Library</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.photoBtn} onPress={handleTakePhoto} activeOpacity={0.8}>
                  <Text style={styles.photoBtnText}>📷 Camera</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>

          {/* Anonymous Toggle */}
          <View style={styles.formGroup}>
            <View style={styles.toggleRow}>
              <View style={styles.toggleLabels}>
                <Text style={styles.label}>🔒 Anonymous Report</Text>
                <Text style={styles.hintText}>Your identity will not be shared with officers</Text>
              </View>
              <Switch
                value={isAnonymous} onValueChange={setIsAnonymous}
                trackColor={{ false: '#334155', true: 'rgba(74,222,128,0.4)' }}
                thumbColor={isAnonymous ? '#4ade80' : '#94a3b8'}
              />
            </View>
          </View>

          {/* Submit */}
          <TouchableOpacity style={[styles.submitBtn, submitting && styles.submitBtnDisabled]} onPress={handleSubmit} disabled={submitting} activeOpacity={0.8}>
            {submitting ? (
              <View style={styles.submittingRow}>
                <ActivityIndicator size="small" color="#ffffff" />
                <Text style={styles.submitBtnText}>  Submitting...</Text>
              </View>
            ) : (
              <Text style={styles.submitBtnText}>Submit Report</Text>
            )}
          </TouchableOpacity>

        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f172a' },
  scrollContent: { padding: 16, paddingBottom: 48 },
  pageHeader: { marginBottom: 20 },
  pageTitle: { fontSize: 22, fontWeight: 'bold', color: '#fff', marginBottom: 6 },
  pageSubtitle: { fontSize: 13, color: '#94a3b8', lineHeight: 20 },
  formCard: { backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)', padding: 18 },
  formGroup: { marginBottom: 22 },
  label: { fontSize: 14, color: '#e2e8f0', fontWeight: '600', marginBottom: 8 },
  required: { color: '#ef4444' },
  hintText: { fontSize: 12, color: '#64748b', marginBottom: 8 },
  errorText: { color: '#f87171', fontSize: 12, marginTop: 4 },
  fieldError: { borderColor: '#ef4444' },
  pickerWrap: { backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 8, borderWidth: 1, borderColor: 'rgba(255,255,255,0.15)', overflow: 'hidden' },
  picker: { color: '#ffffff', height: Platform.OS === 'ios' ? 150 : 52 },
  pickerItem: { color: '#ffffff', fontSize: 14 },
  coordsRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(74,222,128,0.08)', borderRadius: 8, padding: 12, borderWidth: 1, borderColor: 'rgba(74,222,128,0.2)' },
  coordsText: { color: '#4ade80', fontSize: 13, flex: 1 },
  removeLink: { color: '#ef4444', fontSize: 13, fontWeight: '500' },
  locationBtn: { backgroundColor: 'rgba(99,102,241,0.15)', borderRadius: 8, borderWidth: 1, borderColor: 'rgba(99,102,241,0.3)', padding: 13, alignItems: 'center' },
  locationBtnText: { color: '#818cf8', fontWeight: '600', fontSize: 14 },
  dateRow: { flexDirection: 'row', gap: 8 },
  dateBtn: { backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 8, borderWidth: 1, borderColor: 'rgba(255,255,255,0.15)', padding: 12, alignItems: 'center' },
  dateBtnText: { color: '#e2e8f0', fontSize: 13 },
  nowBtn: { backgroundColor: 'rgba(99,102,241,0.2)', borderRadius: 8, borderWidth: 1, borderColor: 'rgba(99,102,241,0.4)', paddingHorizontal: 14, justifyContent: 'center', alignItems: 'center' },
  nowBtnText: { color: '#818cf8', fontWeight: 'bold', fontSize: 13 },
  datePreview: { color: '#4ade80', fontSize: 12, marginTop: 8 },
  textArea: { backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 8, borderWidth: 1, borderColor: 'rgba(255,255,255,0.15)', color: '#fff', padding: 12, fontSize: 14, minHeight: 120, lineHeight: 20 },
  photoButtonsRow: { flexDirection: 'row', gap: 10 },
  photoBtn: { flex: 1, backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 8, borderWidth: 1, borderColor: 'rgba(255,255,255,0.15)', padding: 13, alignItems: 'center' },
  photoBtnText: { color: '#e2e8f0', fontSize: 14, fontWeight: '500' },
  imagePreviewWrap: { position: 'relative', alignSelf: 'flex-start' },
  imagePreview: { width: 220, height: 165, borderRadius: 12, borderWidth: 2, borderColor: 'rgba(255,255,255,0.1)' },
  removeImageBtn: { position: 'absolute', top: -10, right: -10, backgroundColor: '#ef4444', width: 26, height: 26, borderRadius: 13, alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: '#0f172a' },
  removeImageBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 12, lineHeight: 14 },
  toggleRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 12 },
  toggleLabels: { flex: 1 },
  submitBtn: { backgroundColor: '#6366f1', borderRadius: 8, padding: 16, alignItems: 'center', marginTop: 6 },
  submitBtnDisabled: { opacity: 0.6 },
  submittingRow: { flexDirection: 'row', alignItems: 'center' },
  submitBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
});
